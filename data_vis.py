import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import open3d as o3d
from constants.plotting import font
from matplotlib import cm
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.integrate import simps
from tqdm import tqdm
plt.close('all')

# %%
class BeadScan:
    def __init__(self, folderpath, filename, toolname, scan_speed: float,
                 scan_rate: float = 31250, resolution: float = 0.02, max_bead_width: float = 3.0,
                 slice_thickness_override = None, datatrim = (60, 550)):
        # Parameters
        ##############################
        ##############################
        self._folderpath = folderpath  # Path to the folder containing the data file
        self._filename = filename  # Name of the data file
        self._toolname = toolname  # Name of the toolpath file
        self._datatrim = datatrim  # Data trim parameters (leftslice, rightslice)
        self.scan_speed = scan_speed  # Speed of the scan [mm/s]
        self.scan_rate = scan_rate  # Scan rate [Hz]
        self.resolution = resolution  # Resolution of the scan [mm]
        self.max_bead_width = max_bead_width  # Maximum expected bead diameter [mm]

        if slice_thickness_override is not None:
            self.slice_thickness = slice_thickness_override
        else:
            self.slice_thickness = scan_speed / scan_rate  # Thickness of the slice [mm]
        ##############################
        ##############################

        # Load and clean data
        self.time, self.toolpath = self._load_toolpath()  # Load toolpath from file
        self.data_raw = self._load_data()
        self.data_clean = self._clean_data(self.data_raw)
        self.data_trim = self._trim_data(leftslice=self._datatrim[0], rightslice=self._datatrim[1])
        self. X, self.Y, self.Z = self._create_raster()
        self.points, self.valid_mask = self._get_point_cloud()

    def _load_toolpath(self):
        """
        Load the toolpath from a CSV file.
        """
        filepath = f"{self._folderpath}/{self._toolname}"
        toolpath = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1, 2, 3))
        time = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(0,))

        return time, toolpath

    def _load_data(self):
        """
        Load the data from the CSV file.
        """
        filepath = f"{self._folderpath}/{self._filename}"
        data = np.loadtxt(filepath, delimiter=',')

        return data

    def _clean_data(self, data):
        """
        Clean the data by replacing -99999.9999 with 0.0.
        """
        clean_data = np.where(data == -99999.9999, 0.0, data)

        return clean_data

    def _trim_data(self, leftslice=60, rightslice=550):
        """
        Trim the data by slicing the array.
        """

        return self.data_clean[:, rightslice:-leftslice]

    def _create_raster(self):
        """
        Create a raster grid from the data.
        """
        # Assuming the data is in a 2D array format
        rows, cols = self.data_trim.shape
        x = np.arange(cols) * self.resolution
        y = np.arange(rows) * self.resolution
        X, Y = np.meshgrid(x, y)
        Z = self.data_trim

        return X, Y, Z

    def _get_point_cloud(self):
        """
        Convert the data to a point cloud format.
        """
        points = np.column_stack((self.X.ravel(), self.Y.ravel(), self.Z.ravel()))
        valid = self.Z.ravel() > -1e5
        # points_valid = points[valid]

        return points, valid

    def _rotation_matrix_from_vectors(self,vec1, vec2):
        a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s == 0:  # already aligned
            return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-10))

    def flatten_leastsquares(self, visualize=True):
        """
        Flatten the surface using least-squares fitting.
        """
        valid = self.valid_mask
        points = self.points

        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)  # z = C[0]*x + C[1]*y + C[2]

        normal = np.array([-C[0], -C[1], 1.0])
        normal /= np.linalg.norm(normal)
        R = self._rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))
        points_flattened = points @ R.T

        Z_flat = np.full_like(self.Z.ravel(), np.nan)
        Z_flat[valid] = points_flattened[valid, 2]
        Z_flat = Z_flat.reshape(self.Z.shape)
        Z_flat = Z_flat - np.min(Z_flat)  # Normalize to start from zero
        self.Z_flat_leastsquares = Z_flat

        if visualize:
            self._plot_leastsquares()

        return Z_flat, R

    def flatten_ransac(self, visualize=True):
        """
        Flatten the surface using RANSAC plane fitting.
        """
        valid = self.valid_mask
        points = self.points[:, [0, 1]][valid]
        heights = self.points[:, 2][valid]
        # points = np.stack((X_flat[valid_mask], Y_flat[valid_mask]), axis=1)
        # heights = Z_flat[valid_mask]

        # Fit plane using RANSAC: z = a*x + b*y + c
        ransac = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
        ransac.fit(points, heights)

        # Extract coefficients
        coef = ransac.named_steps['ransacregressor'].estimator_.coef_
        intercept = ransac.named_steps['ransacregressor'].estimator_.intercept_
        a, b = coef[1], coef[2]
        c = intercept

        # Define normal vector of plane
        plane_normal = np.array([-a, -b, 1.0])
        plane_normal /= np.linalg.norm(plane_normal)

        # Compute rotation matrix to align plane normal with Z axis
        R = self._rotation_matrix_from_vectors(plane_normal, np.array([0, 0, 1]))

        # Apply rotation to all valid 3D points
        rotated_points = self.points @ R.T
        rotated_points[:,2] = rotated_points[:,2] - c  # Adjust Z by subtracting the intercept
        self.points_flattened = rotated_points

        # Replace X with flattened version
        X_new_flat = np.full_like(self.X.ravel(), np.nan)
        X_new_flat[valid] = rotated_points[valid, 0]
        X_new = X_new_flat.reshape(self.X.shape)
        self.X_flat_ransac = X_new

        # Replace Y with flattened version
        Y_new_flat = np.full_like(self.Y.ravel(), np.nan)
        Y_new_flat[valid] = rotated_points[valid, 1]
        Y_new = Y_new_flat.reshape(self.Y.shape)
        self.Y_flat_ransac = Y_new

        # Replace Z with flattened version
        Z_new_flat = np.full_like(self.Z.ravel(), np.nan)
        Z_new_flat[valid] = rotated_points[valid, 2]
        Z_new = Z_new_flat.reshape(self.Z.shape)
        self.Z_flat_ransac = Z_new

        if visualize:
            self._plot_ransac()

        return Z_new, R

    def _plot_ransac(self):
        """
        Plot the flattened surface.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X_flat_ransac, self.Y_flat_ransac, self.Z_flat_ransac, cmap=cm.viridis)
        ax.set_title("Flattened Ransac")
        # ax.set(xticklabels=[],
        #        yticklabels=[],
        #        zticklabels=[])
        ax.set_aspect('equal')
        # plt.axis('off')
        # plt.grid(b=None)
        plt.show()

    def _plot_leastsquares(self):
        """
        Plot the flattened surface.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X_flat_ransac, self.Y_flat_ransac, self.Z_flat_leastsquares, cmap=cm.viridis)
        ax.set_title("Flattened Least Squares")
        # ax.set(xticklabels=[],
        #        yticklabels=[],
        #        zticklabels=[])
        ax.set_aspect('equal')
        # plt.axis('off')
        # plt.grid(b=None)
        # plt.show()

    def plot_raw(self):
        """
        Plot the raw surface.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.viridis)
        ax.set_title("Raw Surface")
        # ax.set(xticklabels=[],
        #        yticklabels=[],
        #        zticklabels=[])
        ax.set_aspect('equal')
        # plt.axis('off')
        # plt.grid(b=None)
        # plt.show()

    def register_toolpath_to_scan(self, threshold=1.0, visualize=True):
        """
        Registers a 3D toolpath to scanned surface points using ICP.

        Parameters:
            toolpath_xyz : (N, 3) numpy array
                Toolpath point cloud (flattened and transformed).
            scan_xyz : (M, 3) numpy array
                Flattened scan surface as a point cloud (from height map).
            threshold : float
                ICP correspondence distance threshold.
            visualize : bool
                Whether to show before/after alignment using Open3D.

        Returns:
            aligned_toolpath : (N, 3) numpy array
                Toolpath after alignment.
            transformation : (4, 4) numpy array
                Estimated transformation matrix from ICP.
        """

        # Convert to Open3D point clouds
        pcd_toolpath = o3d.geometry.PointCloud()
        pcd_toolpath.points = o3d.utility.Vector3dVector(self.toolpath)

        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(self.points)

        # Initial transform = identity
        init_transform = np.eye(4)

        # Run ICP registration
        reg_result = o3d.pipelines.registration.registration_icp(
            pcd_toolpath, pcd_scan, threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Apply transformation to toolpath
        pcd_toolpath.transform(reg_result.transformation)
        aligned_toolpath = np.asarray(pcd_toolpath.points)

        if visualize:
            print("ICP fitness:", reg_result.fitness)
            print("Transformation matrix:\n", reg_result.transformation)

            pcd_scan.paint_uniform_color([0.6, 0.6, 0.6])  # Green: scan
            pcd_toolpath.paint_uniform_color([1, 0, 0])  # Red: toolpath (aligned)
            o3d.visualization.draw_geometries([pcd_toolpath, pcd_scan])

        return aligned_toolpath, reg_result.transformation

    def extract_profile(self, toolpath_points, scan_points, index, width=0.0, resolution=0.0, visualize=True):
        """
        Extracts an orthogonal slice profile from the scan at a given toolpath index.
        Locates the "noise floor" via RANSAC.
        Identifies where the bead is along the x-direction, assuming only a single profile is present.
        Computes the area under the profile curve via Simpson method.

        Parameters:
            toolpath_points : (N, 3) array
                Aligned toolpath coordinates (after ICP).
            scan_points : (M, 3) array
                Flattened scan point cloud.
            index : int
                Toolpath index to take the slice at.
            width : float
                Half-width of the slice band [mm].
            resolution : float
                Interpolation spacing for profile [mm].
            visualize : bool
                Whether to show the slice profile.

        Returns:
            profile_x : (K,) array
                Distance along slice axis.
            profile_z : (K,) array
                Height values of slice.
            area : float
                Area under the profile curve.
        """
        # Toolpath point and direction

        if resolution <= 0:
            resolution = self.resolution  # Use default resolution if not specified
        if width <= 0:
            width = self.max_bead_width * 2 # Use default width if not specified

        pt = toolpath_points[index]
        if index < len(toolpath_points) - 1:
            direction = toolpath_points[index + 1] - toolpath_points[index]
        else:
            direction = toolpath_points[index] - toolpath_points[index - 1]
        direction /= np.linalg.norm(direction)

        # Orthogonal direction in XY plane
        ortho = np.array([-direction[1], direction[0], 0])
        ortho /= np.linalg.norm(ortho)

        # Project scan points into local coordinates
        rel_points = scan_points - pt
        dist_along_ortho = rel_points @ ortho
        dist_along_dir = rel_points @ direction

        # Select points within band along direction
        mask1 = np.abs(dist_along_ortho) < width
        mask2 = np.abs(dist_along_dir) < self.slice_thickness
        mask = mask1 & mask2
        slice_points = scan_points[mask]


        # if slice_points.size == 0:
        #     print(f"No points found for slice at index {index}")
        #     return None, None

        # Sort by orthogonal axis
        dists = (slice_points - pt) @ ortho
        heights = slice_points[:, 2]

        # Interpolate to uniform spacing
        sort_idx = np.argsort(dists)
        dists_sorted = dists[sort_idx]
        heights_sorted = heights[sort_idx]

        interp = interp1d(dists_sorted, heights_sorted, kind='linear', fill_value="extrapolate")
        profile_x = np.arange(dists_sorted.min(), dists_sorted.max(), resolution)
        profile_z = interp(profile_x)

        if profile_x is None or profile_z is None:
            return 0.0

        # Fit line to lower part of profile using RANSAC
        X = profile_x.reshape(-1, 1)
        y = profile_z
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        line_y = ransac.predict(X)

        # Compute area between profile and fitted line
        area = simps(profile_z - line_y, profile_x)
        area = max(area, 0.0)

        if visualize:
            print(len(profile_x), "points in profile")
            print(f"Computed area: {area:.3f} mm^2")

            self._plot_profile_area(profile_x, profile_z, index=index, ransac_line=line_y)


            # # plot the slice points on top of the scan points plot using o3d
            # pcd_slice = o3d.geometry.PointCloud()
            # pcd_slice.points = o3d.utility.Vector3dVector(slice_points)
            # pcd_slice.paint_uniform_color([1, 0, 0])  # Red for slice points
            # pcd_scan = o3d.geometry.PointCloud()
            # pcd_scan.points = o3d.utility.Vector3dVector(scan_points)
            # pcd_scan.paint_uniform_color([0.6, 0.6, 0.6])
            # o3d.visualization.draw_geometries([pcd_scan, pcd_slice])

            self._plot_profile_search_region(slice_points, scan_points)


        return profile_x, profile_z, line_y, area

    # def get_profile_area(self, profile_x, profile_z, visualize=False):
    #     """
    #     Locate the "noise floor" via RANSAC. Identify where the bead is along the x-direction,
    #     assuming only a single profile is present. Compute the area under the profile curve via Simpson method.
    #
    #     Parameters:
    #         profile_x : (K,) array
    #             Distance along slice axis.
    #         profile_z : (K,) array
    #             Height values of slice.
    #
    #     Returns:
    #         area : float
    #             Area under the profile curve.
    #     """
    #     if profile_x is None or profile_z is None:
    #         return 0.0
    #
    #     # Fit line to lower part of profile using RANSAC
    #     X = profile_x.reshape(-1, 1)
    #     y = profile_z
    #     ransac = RANSACRegressor()
    #     ransac.fit(X, y)
    #     line_y = ransac.predict(X)
    #
    #     # Compute area between profile and fitted line
    #     area = simps(profile_z - line_y, profile_x)
    #     area = max(area, 0.0)
    #
    #     if visualize:
    #         print(f"Computed area: {area:.3f} mm^2")
    #
    #         self._plot_profile_area(profile_x, profile_z, ransac_line=line_y)
    #
    #     return area

    def _plot_profile_area(self, profile_x, profile_z, index=None, ransac_line=None):
        """
        Plot the profile and shaded area under the curve.

        Parameters:
            profile_x : (K,) array
                Distance along slice axis.
            profile_z : (K,) array
                Height values of slice.
            area : float
                Area under the profile curve.
        """
        plt.figure()
        plt.plot(profile_x, profile_z, '-k', label='Profile')
        if ransac_line is not None:
            plt.plot(profile_x, ransac_line, '--r', label='RANSAC floor')
            plt.fill_between(profile_x, ransac_line, profile_z, color='gray', alpha=0.5, label='Area')
        plt.xlabel("Distance along slice [mm]")
        plt.ylabel("Height [mm]")
        if index is not None:
            plt.title(f"Profile Slice at Toolpath Index {index}")
        else:
            plt.title(f"Profile at Toolpath Slice")
        plt.grid(True)
        plt.legend()
        plt.show()

    def _plot_profile_search_region(self, slice_points, scan_points):
        # plot the slice points on top of the scan points plot using o3d
        pcd_slice = o3d.geometry.PointCloud()
        pcd_slice.points = o3d.utility.Vector3dVector(slice_points)
        pcd_slice.paint_uniform_color([1, 0, 0])  # Red for slice points
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan_points)
        pcd_scan.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([pcd_scan, pcd_slice])

    def get_all_profile_areas(self, toolpath_points, scan_points, width=0.0, resolution=0.0, visualize=False):
        """
        Compute the area under the profile curve for each toolpath index.

        Parameters:
            toolpath_points : (N, 3) array
                Aligned toolpath coordinates (after ICP).
            scan_points : (M, 3) array
                Flattened scan point cloud.
            width : float
                Half-width of the slice band [mm].
            resolution : float
                Interpolation spacing for profile [mm].
            visualize : bool
                Whether to show the slice profiles.

        Returns:
            areas : list of float
                List of areas under the profile curves for each toolpath index.
        """
        # generate a profile_x, profile_z, and area for each toolpath index, and do a lil status bar in the console
        profile_xs = []
        profile_zs = []
        ransac_lines = []
        areas = []

        for i in tqdm(range(len(toolpath_points)), desc="Extracting profiles"):
            px, pz, rl, a = beadscan.extract_profile(toolpath_points, scan_points, index=i, visualize=False)
            profile_xs.append(px)
            profile_zs.append(pz)
            ransac_lines.append(rl)
            areas.append(a)

        if visualize:
            for i in range(len(toolpath_points)):
                px = profile_xs[i]
                pz = profile_zs[i]
                rl = ransac_lines[i]
                if px is None or pz is None:
                    print(f"No profile found for toolpath index {i}, skipping visualization.")
                    continue

                # plt.figure(1)
                # plt.plot(px, pz, '-k')
                # plt.xlabel("Distance along slice [mm]")
                # plt.ylabel("Height [mm]")
                # plt.title(f"Orthogonal slice at toolpath index {i}, area={areas[i]:.3f} mm^2")
                # plt.grid(True)
                # plt.show()
                # plt.pause(0.1)
                # plt.close(1)

                self._plot_profile_area(px, pz, index=i, ransac_line=rl)
                plt.pause(0.1)
                plt.close()
        plt.pause(0.1)

        return areas, profile_xs, profile_zs

    def get_flowrates(self, areas, visualize=False):
        """
        Compute the flow rate as a function of time, based on the areas and bead slice thickness.

        Parameters:
            areas : list of float
                List of areas under the profile curves for each toolpath index.
            scan_speed : float
                Speed of the scan [mm/s]. If None, uses self.scan_speed.

        Returns:
            flowrate : float
                Flow rate in mm^3/s.
        """
        # check that self.time is constant intervals (eg, 0.1s steps)
        t = self.time
        test0 = np.round(t[1:] - t[:-1], 6)
        dt = np.mean(test0)
        test1 = test0 == dt  # check if all time steps are equal
        if not test1.all():
            print("Warning. Hard check failed! Inconsistent time steps detected")
        else:
            print("Hard check passed: Consistent time steps is completely enforced")
        test2 = test0 - dt < 0.01 * dt
        test3 = np.all(test2)
        assert test3, "Soft check also failed! Time steps are not consistent enough"
        print("Soft check passed: Consistent time steps is mostly enforced")\

        dt = np.round(np.diff(t,prepend=np.mean(t)),6)  # s

        volumes = np.array(areas) * self.slice_thickness  # mm^3
        flowrates = volumes / dt  # mm^3/s

        if visualize:
            self._plot_flowrates(flowrates, volumes)

        return flowrates, volumes

    def _plot_flowrates(self, flowrates, volumes):
        """
        Plot the flow rates and volumes over time.

        Parameters:
            flowrates : list of float
                Flow rates in mm^3/s.
            volumes : list of float
                Volumes in mm^3.s
        """
        t = self.time
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, flowrates, label='Flow Rate (mm^3/s)', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Flow Rate (mm^3/s)')
        plt.title('Flow Rate Over Time')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, volumes, label='Volume (mm^3)', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Volume (mm^3)')
        plt.title('Volume Over Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()




'''    # todo: testing
    def compute_toolpath_tangents(self, toolpath):
        """Compute unit tangents for each toolpath index using central differences."""
        n = len(toolpath)
        tangents = np.zeros_like(toolpath)
        for i in range(n):
            if 0 < i < n - 1:
                vec = toolpath[i + 1] - toolpath[i - 1]
            elif i == 0:
                vec = toolpath[1] - toolpath[0]
            else:
                vec = toolpath[-1] - toolpath[-2]
            norm = np.linalg.norm(vec)
            tangents[i] = vec / (norm + 1e-12)
        return tangents

    def make_plane_basis(self, t):
        """
        Given unit tangent t (plane normal), return two orthonormal vectors u, v that span the plane.
        Similar to previous helper but as a method.
        """
        z = np.array([0.0, 0.0, 1.0])
        u = np.cross(t, z)
        if np.linalg.norm(u) < 1e-8:
            ref = np.array([0.0, 1.0, 0.0])
            u = np.cross(t, ref)
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        v /= np.linalg.norm(v)
        return u, v

    def _profile_from_slab(self, slab_points, center, u_axis, v_axis, num_bins=200, min_valid_bins=3):
        """
        Build a 1D lateral profile from slab_points projected into (u,v) coordinates.
        Returns dict with u_bins, v_profile, area, proj_u, proj_v, n_points
        """
        if slab_points is None or slab_points.shape[0] == 0:
            return None

        rel = slab_points - center
        proj_u = rel @ u_axis
        proj_v = rel @ v_axis

        u_min, u_max = proj_u.min(), proj_u.max()
        if u_max == u_min:
            return None

        bins = np.linspace(u_min, u_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        bin_idx = np.digitize(proj_u, bins) - 1
        max_v = np.full(num_bins, np.nan)
        counts = np.zeros(num_bins, dtype=int)

        for k in range(num_bins):
            mask = bin_idx == k
            if np.any(mask):
                max_v[k] = np.nanmax(proj_v[mask])
                counts[k] = np.count_nonzero(mask)

        valid = ~np.isnan(max_v)
        if np.sum(valid) < min_valid_bins:
            return None

        interp = interp1d(bin_centers[valid], max_v[valid], kind='linear', bounds_error=False, fill_value=0.0)
        u_uniform = bin_centers
        v_uniform = interp(u_uniform)
        area = np.trapz(v_uniform, u_uniform)

        return {
            'u_bins': u_uniform,
            'v_profile': v_uniform,
            'area': area,
            'proj_u': proj_u,
            'proj_v': proj_v,
            'n_points': slab_points.shape[0],
            'bin_counts': counts
        }

    def compute_all_slices_vectorized(self,
                                      toolpath_points,
                                      scan_points=None,
                                      slab_half_width=0.25,
                                      num_bins=200,
                                      min_points=20,
                                      chunk_size=200,
                                      resolution=None):
        """
        Compute orthogonal slices for ALL toolpath points using a vectorized chunked approach.

        Parameters:
            toolpath_points : (N,3) numpy array - aligned toolpath (after ICP)
            scan_points : (M,3) numpy array - flattened scan points; if None, uses self.points_flattened
            slab_half_width : float - half-width of the slicing slab (same units as scan)
            num_bins : int - lateral bins for building profile (max-per-bin)
            min_points : int - minimum slab points required to build a profile
            chunk_size : int - number of toolpath slices handled at once to limit memory
            resolution : float or None - optional lateral sampling step (not needed, bins set it)
        Returns:
            num_computed : int - number of successfully computed slices
        Side effects:
            sets self.slices = list(len(toolpath_points)) of slice dicts or None
        """
        if scan_points is None:
            if not hasattr(self, 'points_flattened'):
                raise RuntimeError(
                    "scan_points not provided and self.points_flattened doesn't exist. Run flatten_ransac() first.")
            scan_points = self.points_flattened.copy()

        N = len(toolpath_points)
        M = scan_points.shape[0]
        tangents = self.compute_toolpath_tangents(toolpath_points)

        # Prepare storage
        self.slices = [None] * N

        # chunk over toolpath indices
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            k = end - start
            tp_chunk = toolpath_points[start:end]  # (k,3)
            t_chunk = tangents[start:end]  # (k,3)

            # vectorized difference: shape (k, M, 3)
            vecs = scan_points[np.newaxis, :, :] - tp_chunk[:, np.newaxis, :]
            # distances to plane (signed) shape (k, M)
            dists = np.einsum('kmi,kmi->km', vecs, t_chunk[:, np.newaxis, :])
            slab_mask = np.abs(dists) <= slab_half_width  # (k, M) boolean

            # For each slice in chunk, extract slab points and build profile
            for local_idx in range(k):
                global_idx = start + local_idx
                mask = slab_mask[local_idx]
                if np.count_nonzero(mask) < min_points:
                    # not enough points
                    self.slices[global_idx] = None
                    continue

                slab_points = scan_points[mask]  # (S,3)

                # build basis u,v for this tangent
                t = t_chunk[local_idx]
                u_axis, v_axis = self.make_plane_basis(t)

                profile = self._profile_from_slab(slab_points, tp_chunk[local_idx], u_axis, v_axis,
                                                  num_bins=num_bins)
                if profile is None:
                    self.slices[global_idx] = None
                    continue

                # pack metadata
                profile.update({
                    'center': tp_chunk[local_idx].copy(),
                    'tangent': t.copy(),
                    'u_axis': u_axis.copy(),
                    'v_axis': v_axis.copy()
                })
                self.slices[global_idx] = profile

        num_computed = sum(1 for s in self.slices if s is not None)
        print(f"Computed {num_computed}/{N} slices (chunk_size={chunk_size}).")
        return num_computed'''


if __name__ == "__main__":
    # Fake surface with slight tilt + random "blob"
    # nx, ny = 100, 120
    # X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    # Z = 0.01 * X + 0.02 * Y + np.random.normal(scale=0.01, size=(nx, ny))

    # datatable = np.loadtxt('data/Silicone bead data.csv', delimiter=',')

    # # Add an outlier "bead" to simulate scan artifact
    # Z[50:55, 60:65] += 0.5
    # Z[30:32, 40:42] = -99999.9999  # Simulate invalid region

    # Z_flat, _ = flatten_surface(Z)

    # Plot comparison
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(X, Y, Z, cmap='viridis')
    # ax1.set_title("Original Warped Surface")
    #
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot_surface(X, Y, Z_flat, cmap='viridis')
    # ax2.set_title("Flattened Surface")
    #
    # plt.tight_layout()
    # plt.show()

    folderpath = 'data'
    filename = 'Silicone bead data.csv'
    toolname = 'bead_toolpath.csv'
    scan_speed = 10.0  # mm/s

    # toolpath =

    beadscan = BeadScan(folderpath, filename, toolname, scan_speed, slice_thickness_override=0.05)
    Z_rs, R_rs = beadscan.flatten_ransac(visualize=True)
    beadscan.plot_raw()
    toolpath_aligned, toolpath_transform = beadscan.register_toolpath_to_scan(visualize=True)
    scan_points = beadscan.points_flattened

    areas, profile_xs, profile_zs = beadscan.get_all_profile_areas(toolpath_aligned, scan_points, visualize=True)

    flowrates, volumes = beadscan.get_flowrates(areas, visualize=True)

    profile_x, profile_z, ransac_line, area = beadscan.extract_profile(toolpath_aligned, scan_points,
                                                    index=123, width=0.0, visualize=True)
    # area = beadscan.get_profile_area(profile_x, profile_z visualize=True)










    # # Vectorized compute all slices (tune slab_half_width, num_bins, chunk_size)
    # num = beadscan.compute_all_slices_vectorized(toolpath_aligned,
    #                                              scan_points=scan_points,
    #                                              slab_half_width=0.5,
    #                                              num_bins=200,
    #                                              min_points=20,
    #                                              chunk_size=300)  # adjust chunk_size to suit memory

