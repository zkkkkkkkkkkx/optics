import sys
import os
import numpy as np
import math
from optitrace.system import OpticalSystem
from optitrace.optical import (
    ImagePlane,
    DummyPlane,
    IdealMirror,
    IdealThinLens,
    OpticalSource,
)
from optitrace.core import Vector3D, Transform3D, Point3D
from optitrace.visualization import InteractiveRenderer

def _xyz_tuple(obj):

    if hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z"):
        return (float(obj.x), float(obj.y), float(obj.z))
    if isinstance(obj, (tuple, list, np.ndarray)) and len(obj) >= 3:
        return (float(obj[0]), float(obj[1]), float(obj[2]))
    raise TypeError(f"Unsupported xyz object type: {type(obj)}")


class Receiver_OpticalPathModel:

    def __init__(self):
        self.atp: OpticalSystem | None = None
        self._mirror_center = Point3D(0.0, 0.0, 40.0)  # gimbal位置
        self._source_center = Point3D(0.0, 0.0, 0.0)
        self._source_direction = Vector3D(0.0, 0.0, 1.0)

    @staticmethod
    def _mrad_to_rad(value_mrad: float) -> float:
        return float(value_mrad) * 1e-3

    @staticmethod
    def _extract_hit_point(surface_obj):
        hit_points = getattr(surface_obj, "hit_point", None)
        if not hit_points:
            return None
        hit = hit_points[0]
        return _xyz_tuple(hit)

    def _debug_receive_path_status(self):
        comps = self.atp.path_config_components.get("tracking", [])
        if not comps:
            print("[ReceiverDebug] receive path is empty.")
            return
        print("[ReceiverDebug] receive path hit status:")
        for idx, comp in enumerate(comps):
            surf = comp.surfaces[0]
            name = getattr(comp, "name", getattr(surf, "name", f"surface_{idx}"))
            hit_points = getattr(surf, "hit_point", None)
            hit_count = len(hit_points) if hit_points else 0
            print(f"  idx={idx:02d}, name={name}, hit_count={hit_count}")

    def set_angles(self, emit_direction, baijing_fuyang, baijing_fangwei, FSM_fuyang, FSM_fangwei):
        gimbal_offset = Vector3D(
            self._mrad_to_rad(baijing_fuyang),
            self._mrad_to_rad(baijing_fangwei),
            0.0,
        )
        fsm1_offset = Vector3D(
            self._mrad_to_rad(FSM_fuyang),
            self._mrad_to_rad(FSM_fangwei),
            0.0,
        )

        # print('[DEBUG]:',emit_direction)
        if abs(float(emit_direction[2])) < 1e-12:
            raise ValueError("emit_direction.z不能为0，否则无法保证source_center的z为0")

        s = float(self._mirror_center.z) / float(emit_direction[2])
        self._source_center = Point3D(
            float(self._mirror_center.x) - float(emit_direction[0]) * s,
            float(self._mirror_center.y) - float(emit_direction[1]) * s,
            0.0,
        )
        self._source_direction = Vector3D(
            float(emit_direction[0]),
            float(emit_direction[1]),
            float(emit_direction[2])
        )
        self.atp = build_atp_v0(gimbal_offset=gimbal_offset, fsm1_offset=fsm1_offset)

    def compute_location(self, chief_only: bool = False, VISUAL: bool = False, debug_on_fail: bool = True):
        if self.atp is None:
            self.set_angles(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        print("source_direction: [{:.8f}, {:.8f}, {:.8f}]".format(
            float(self._source_direction.x), float(self._source_direction.y), float(self._source_direction.z)))
        print("gimbal_offset: [{:.8f}, {:.8f}, {:.8f}]".format(
            float(self.atp.gimbal_offset.x), float(self.atp.gimbal_offset.y), float(self.atp.gimbal_offset.z)))
        print("fsm1_offset: [{:.8f}, {:.8f}, {:.8f}]".format(
            float(self.atp.fsm1_offset.x), float(self.atp.fsm1_offset.y), float(self.atp.fsm1_offset.z)))
        print("fsm2_offset: [{:.8f}, {:.8f}, {:.8f}]".format(
            float(self.atp.fsm2_offset.x), float(self.atp.fsm2_offset.y), float(self.atp.fsm2_offset.z)))

        num_rays = 1 if chief_only else 19
        rays = OpticalSource.array_source_hexagonal(
            self._source_center,
            self._source_direction,
            10.0,
            num_rays,
            wavelength=1550.0,
        )
        self.atp.trace_ray("tracking", rays)
        last_surface = self.atp.path_config_components["tracking"][-1].surfaces[0]
        center = self._extract_hit_point(last_surface)
        if center is None:
            if debug_on_fail:
                self._debug_receive_path_status()

            raise ValueError("没有打在最后一个平面")

        if VISUAL:
            renderer = InteractiveRenderer(window_name="tracking")
            renderer.add_optical_system(self.atp, "tracking")
            renderer.start_interactor()
        return [center[0] - last_surface.center.x, center[1] - last_surface.center.y, center[2] - last_surface.center.z]

    def get_location(
            self,
            emit_direction,
            baijing_fuyang,
            baijing_fangwei,
            FSM_fuyang,
            FSM_fangwei,
            chief_only: bool = False,
            debug_on_fail: bool = True,
    ):
        self.set_angles(emit_direction, baijing_fuyang, baijing_fangwei, FSM_fuyang, FSM_fangwei)
        return self.compute_location(chief_only=chief_only, debug_on_fail=debug_on_fail)


class Transmitter_OpticalPathModel:

    def __init__(self):
        self.atp: OpticalSystem | None = None
        self._source_center = Point3D(0.0, -179.2820323027556, 40.0)
        self._source_direction = Vector3D(0.0, 0.5000000000000024, -0.8660254037844373).normalize()

    @staticmethod
    def _mrad_to_rad(value_mrad: float) -> float:
        return float(value_mrad) * 1e-3

    @staticmethod
    def _extract_ray_direction(surface_obj):
        hit_rays = getattr(surface_obj, "_hit_ray", None)
        if not hit_rays:
            return None
        direction = hit_rays[0].current_direction
        return _xyz_tuple(direction)

    def set_angles(
            self,
            tilting_mirror_pitch_mrad: float,
            tilting_mirror_azimuth_mrad: float,
            fsm1_pitch_mrad: float,
            fsm1_azimuth_mrad: float,
            fsm2_pitch_mrad: float,
            fsm2_azimuth_mrad: float,
    ) -> None:
        gimbal_offset = Vector3D(
            self._mrad_to_rad(tilting_mirror_pitch_mrad),
            self._mrad_to_rad(tilting_mirror_azimuth_mrad),
            0.0,
        )
        fsm1_offset = Vector3D(
            self._mrad_to_rad(fsm1_pitch_mrad),
            self._mrad_to_rad(fsm1_azimuth_mrad),
            0.0,
        )
        fsm2_offset = Vector3D(
            self._mrad_to_rad(fsm2_pitch_mrad),
            self._mrad_to_rad(fsm2_azimuth_mrad),
            0.0,
        )
        self.atp = build_atp_v0(
            gimbal_offset=gimbal_offset,
            fsm1_offset=fsm1_offset,
            fsm2_offset=fsm2_offset,
        )

    def compute_pointing_vector(self, VISUAL: bool = False) -> tuple[float, float, float]:
        if self.atp is None:
            self.set_angles(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        rays = OpticalSource.array_source_hexagonal(
            self._source_center,
            self._source_direction,
            1.0,
            1,
            wavelength=1550.0,
        )
        self.atp.trace_ray("transmit", rays)
        last_surface = self.atp.path_config_components["transmit"][-1].surfaces[0]
        vec = self._extract_ray_direction(last_surface)
        if vec is None:
            return (0.0, 0.0, 0.0)

        d_vec = np.array(vec, dtype=float)
        norm = float(np.linalg.norm(d_vec))
        if norm < 1e-12:
            return (0.0, 0.0, 0.0)
        d_out = d_vec / norm

        if VISUAL:
            renderer = InteractiveRenderer(window_name="transmit")
            renderer.add_optical_system(self.atp, "transmit")
            renderer.start_interactor()
        return (float(d_out[0]), float(d_out[1]), float(d_out[2]))

    def get_vector(
            self,
            tilting_mirror_pitch_mrad: float,
            tilting_mirror_azimuth_mrad: float,
            fsm1_pitch_mrad: float,
            fsm1_azimuth_mrad: float,
            fsm2_pitch_mrad: float,
            fsm2_azimuth_mrad: float,
    ) -> tuple[float, float, float]:
        self.set_angles(
            tilting_mirror_pitch_mrad,
            tilting_mirror_azimuth_mrad,
            fsm1_pitch_mrad,
            fsm1_azimuth_mrad,
            fsm2_pitch_mrad,
            fsm2_azimuth_mrad,
        )
        return self.compute_pointing_vector()


def build_atp_v0(
        gimbal_offset=Vector3D(0, 0, 0),
        fsm1_offset=Vector3D(0, 0, 0),
        fsm2_offset=Vector3D(0, 0, 0),
) -> OpticalSystem:

    atp = OpticalSystem()
    atp.gimbal_offset = gimbal_offset
    atp.fsm1_offset = fsm1_offset
    atp.fsm2_offset = fsm2_offset
    image_transmit = ImagePlane(
        center=Point3D(0, 0, 0),
        normal=Transform3D.rotate_nomarl(0, 0, 0),
        aperture=300.0,
        name="image_transmit",
        description="image_transmit plane of posterior path, for transmit subsystem",
    )
    gimbal_mirror = IdealMirror(
        center=Point3D(0, 0, 40),
        normal=Transform3D.rotate_nomarl(
            -math.pi / 4 + gimbal_offset.x, gimbal_offset.y, gimbal_offset.z
        ),
        aperture=84.84,
        name="gimbal_mirror",
    )
    len1_offaxis = IdealThinLens(
        center=Point3D(0, -50, 40),
        normal=Transform3D.rotate_nomarl(-math.pi / 2, 0, 0),
        aperture=62.0,
        focal_length=50.0,
        name="len1_offaxis",
        description="Alternative Off-Axis Double-Reflection System,part1",
    )
    len2_offaxis = IdealThinLens(
        center=Point3D(0, -95, 40),
        normal=Transform3D.rotate_nomarl(-math.pi / 2, 0, 0),
        aperture=7.0,
        focal_length=-5.0,
        name="len2_offaxis",
        description="Alternative Off-Axis Double-Reflection System,part2",
    )
    mirror_offaxis = IdealMirror(
        center=Point3D(0, -110, 40),
        normal=Transform3D.rotate_nomarl(math.pi * 1 / 6, 0, 0),
        aperture=10.0,
        name="mirror_offaxis",
        description="Alternative Off-Axis Double-Reflection System,part3",
    )
    mirror_fsm1 = IdealMirror(
        center=Point3D(0, -127.5 + 0.17949192431105132, 10),
        normal=Transform3D.rotate_nomarl(fsm1_offset.x, fsm1_offset.y, fsm1_offset.z),
        aperture=10.0,
        name="mirror_fsm1",
    )
    wdm_1 = IdealMirror(
        center=Point3D(0, -143 - 1.6410161513778405, 40),
        normal=Transform3D.rotate_nomarl(0, 0, 0),
        aperture=10.0,
        name="wdm_1",
        description="WDM1 mirror between FSM1 and FSM2, for transmit subsystem",  # noqa: E501
    )
    dummy_plane1 = DummyPlane(
        center=wdm_1.center,
        normal=wdm_1.normal,
        aperture=10.0,
        name="dummy_plane",
        description="Dummy plane wdm_1 between FSM1 and FSM2, for tracking subsystem and receive subsystem",
        # noqa: E501
    )

    mirror_fsm2 = IdealMirror(
        center=Point3D(0, -164 + 2.038475772933282, 10),
        normal=Transform3D.rotate_nomarl(fsm2_offset.x, fsm2_offset.y, fsm2_offset.z),
        aperture=10.0,
        name="mirror_fsm2",
        description="mirror_fsm2, for transmit subsystem",
    )

    wdm_2 = IdealMirror(
        center=Point3D(0, -164 - 1.290163190291878, 78 - 2.2346281956406813),
        normal=Transform3D.rotate_nomarl(math.pi * 1 / 3, 0, 0),
        aperture=10.0,
        name="wdm_2",
        description="WDM2 mirror, for tracking subsystem",
    )
    dummy_plane2 = DummyPlane(
        center=wdm_2.center,
        normal=wdm_2.normal,
        aperture=10.0,
        name="dummy_plane2",
        description="Dummy plane wdm_2 between mirror_receive, for tracking subsystem",
    )

    len_tracking = IdealThinLens(
        center=Point3D(0, -140, 78 - 2.2346281956407665),
        normal=Transform3D.rotate_nomarl(-math.pi / 2, 0, 0),
        aperture=10.0,
        focal_length=65.0,
        name="len_tracking",
        description="Tracking lens of posterior path, for tracking subsystem",
    )
    image_tracking = ImagePlane(
        center=Point3D(0, -75, 78 - 2.2346281956407665),
        normal=Transform3D.rotate_nomarl(-math.pi / 2, 0, 0),
        aperture=15.0,
        name="image_tracking",
        description="Tracking image plane of posterior path, for tracking subsystem",
    )

    mirror_receive = IdealMirror(
        center=Point3D(0, -180 - 1.9728101671130673, 108 - 3.3394360270512493),
        normal=Transform3D.rotate_nomarl(math.pi * 1 / 3, 0, 0),
        aperture=10.0,
        name="mirror_receive",
        description="mirror_receive, for receive subsystem",
    )
    len_receive = IdealThinLens(
        center=Point3D(0, -150, 108 - 3.339436027051349),
        normal=Transform3D.rotate_nomarl(-math.pi / 2, 0, 0),
        aperture=10.0,
        focal_length=50.0,
        name="len_receive",
        description="len_receive, for receive subsystem",
    )
    image_receive = ImagePlane(
        center=Point3D(0, -100, 108 - 3.339436027051505),
        normal=Transform3D.rotate_nomarl(-math.pi / 2, 0, 0),
        aperture=15.0,
        name="image_receive",
        description="image_receive plane of posterior path, for receive subsystem",
    )

    image_transmit_reverse = ImagePlane(
        center=Point3D(0, -180 + 0.7179676972444042, 40),
        normal=Transform3D.rotate_nomarl(0, 0, 0),
        aperture=30.0,
        name="image_transmit_reverse",
        description="image_transmit_reverse plane of posterior path, for transmit subsystem",
    )

    atp.configure_path_components(
        "tracking",
        [
            gimbal_mirror,
            len1_offaxis,
            len2_offaxis,
            mirror_offaxis,
            mirror_fsm1,
            dummy_plane1,
            wdm_2,
            len_tracking,
            image_tracking,
        ],
    )
    atp.configure_path_components(
        "receive",
        [
            gimbal_mirror,
            len1_offaxis,
            len2_offaxis,
            mirror_offaxis,
            mirror_fsm1,
            dummy_plane1,
            dummy_plane2,
            mirror_receive,
            len_receive,
            image_receive,
        ],
    )
    atp.configure_path_components(
        "transmit",
        [
            mirror_fsm2,
            wdm_1,
            mirror_fsm1,
            mirror_offaxis,
            len2_offaxis,
            len1_offaxis,
            gimbal_mirror,
            image_transmit,
        ],
    )
    atp.configure_path_components(
        "transmit_reverse",
        atp.path_config_components["transmit"][::-1][1:],
    )
    atp.path_config_components["transmit_reverse"].append(image_transmit_reverse)

    return atp


def trace_tracking_scene(
        atp: OpticalSystem,
        source_center: Point3D,
        source_direction: Vector3D,
        aperture=10.0,
        num_rays=19,
):
    position = source_center
    direction = source_direction.normalize()
    rays = OpticalSource.array_source_hexagonal(
        position, direction, aperture, num_rays, wavelength=950
    )
    atp.trace_ray("tracking", rays)

    surface_idx = 8
    # atp.path_config_components["tracking"][surface_idx].surfaces[0].plot_hit_spot(size=0.05, window_name="tracking")
    hitpoint = (
        atp.path_config_components["tracking"][surface_idx].surfaces[0].hit_point[0]
    )
    targetpoint = atp.path_config_components["tracking"][surface_idx].surfaces[0].center
    print(
        f"tracking hit point: {hitpoint}\n"
        + f"surface center: {targetpoint}\n"
        + f"error={hitpoint - targetpoint}"
    )

    renderer = InteractiveRenderer(window_name="tracking")
    renderer.add_optical_system(atp, "tracking")
    renderer.start_interactor()
    pass


def trace_receive_scene(
        atp: OpticalSystem,
        source_center: Point3D,
        source_direction: Vector3D,
        aperture=10.0,
        num_rays=19,
):
    position = source_center
    direction = source_direction.normalize()
    rays = OpticalSource.array_source_hexagonal(
        position, direction, aperture, num_rays, wavelength=1550.0
    )
    atp.trace_ray("receive", rays)

    surface_idx = 9
    # atp.path_config_components["receive"][surface_idx].surfaces[0].plot_hit_spot(size=0.05, window_name="receive")
    hitpoint = (
        atp.path_config_components["receive"][surface_idx].surfaces[0].hit_point[0]
    )
    targetpoint = atp.path_config_components["receive"][surface_idx].surfaces[0].center
    print(
        f"receive hit point: {hitpoint}\n"
        + f"surface center: {targetpoint}\n"
        + f"error={hitpoint - targetpoint}"
    )

    renderer = InteractiveRenderer(window_name="receive")
    renderer.add_optical_system(atp, "receive")
    renderer.start_interactor()
    pass


def trace_transmit_reverse_scene(
        atp: OpticalSystem,
        source_center: Point3D,
        source_direction: Vector3D,
        aperture=10.0,
        num_rays=19,
):
    position = source_center
    direction = source_direction.normalize()
    rays = OpticalSource.array_source_hexagonal(
        position, direction, aperture, num_rays, wavelength=1550.0
    )
    atp.trace_ray("transmit_reverse", rays)

    surface_idx = 7
    # atp.path_config_components["transmit_reverse"][surface_idx].surfaces[0].plot_hit_spot(size=0.05, window_name="transmit_reverse")
    hitpoint = (
        atp.path_config_components["transmit_reverse"][surface_idx]
        .surfaces[0]
        .hit_point[0]
    )
    targetpoint = (
        atp.path_config_components["transmit_reverse"][surface_idx].surfaces[0].center
    )
    print(
        f"transmit_reverse hit point: {hitpoint}\n"
        + f"surface center: {targetpoint}\n"
        + f"error={hitpoint - targetpoint}"
    )
    direction = (
        atp.path_config_components["transmit_reverse"][-1]
        .surfaces[0]
        ._hit_ray[0]
        .current_direction
    )
    print(f"ray direction: {direction}")
    renderer = InteractiveRenderer(window_name="transmit_reverse")
    renderer.add_optical_system(atp, "transmit_reverse")
    renderer.start_interactor()
    pass


def trace_transmit_scene(
        atp: OpticalSystem,
        source_center: Point3D,
        source_direction: Vector3D,
        aperture=1.0,
        num_rays=19,
):
    position = source_center
    direction = source_direction.normalize()
    rays = OpticalSource.array_source_hexagonal(
        position, direction, aperture, num_rays, wavelength=1550.0
    )
    atp.trace_ray("transmit", rays)

    surface_idx = 7
    # atp.path_config_components["transmit"][surface_idx].surfaces[0].plot_hit_spot(size=0.05, window_name="transmit")
    hitpoint = (
        atp.path_config_components["transmit"][surface_idx].surfaces[0].hit_point[0]
    )
    targetpoint = atp.path_config_components["transmit"][surface_idx].surfaces[0].center
    print(
        f"transmit hit point: {hitpoint}\n"
        + f"surface center: {targetpoint}\n"
        + f"error={hitpoint - targetpoint}"
    )

    renderer = InteractiveRenderer(window_name="transmit")
    renderer.add_optical_system(atp, "transmit")
    renderer.start_interactor()
    pass

    # INSERT_YOUR_CODE
    # === 调用Receiver_OpticalPathModel并在初始状态下输出ccd的位置 ===


import csv
import os
import math
import numpy as np

if __name__ == "__main__":
    recv = Receiver_OpticalPathModel()
    recv.set_angles(
        emit_direction=(0.0, 0.0, 1.0),
        baijing_fuyang=0.0,
        baijing_fangwei=0.0,
        FSM_fuyang=0.0,
        FSM_fangwei=0.0,
    )
    base_ccd = recv.compute_location(chief_only=True, VISUAL=False)
    print(f"Baseline CCD position (all angles zero): {base_ccd}")

    output_dir = r"./trackval"
    os.makedirs(output_dir, exist_ok=True)

    # gimbal偏转：俯仰从0到0.0015度，每步0.0001
    csv_path_gimbal_fuyang = os.path.join(output_dir, "trackval_gimbal_fuyang.csv")
    gimbal_fuyang_degs = [round(x, 4) for x in np.arange(0.0, 0.0016, 0.0001)]  # 0, ..., 0.0015
    gimbal_fuyang_rads = [x * 1000 * math.pi / 180 for x in gimbal_fuyang_degs]  # mrad

    with open(csv_path_gimbal_fuyang, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["baijing_fuyang(deg)", "baijing_fuyang(rad)", "CCD_x", "CCD_y", "CCD_z", "Dist_from_origin"])
        for deg, mrad in zip(gimbal_fuyang_degs, gimbal_fuyang_rads):
            recv.set_angles(
                emit_direction=(0.0, 0.0, 1.0),
                baijing_fuyang=mrad,
                baijing_fangwei=0.0,
                FSM_fuyang=0.0,
                FSM_fangwei=0.0,
            )
            ccd_loc = recv.compute_location(chief_only=True, VISUAL=False)
            d = math.sqrt(
                (ccd_loc[0] - base_ccd[0]) ** 2 +
                (ccd_loc[1] - base_ccd[1]) ** 2 +
                (ccd_loc[2] - base_ccd[2]) ** 2
            )
            print(f"baijing_fuyang={deg:.4f} deg ({mrad:.8f} rad), CCD={ccd_loc}, Δ={d}")
            writer.writerow([deg, mrad] + list(ccd_loc) + [d])
    print(f"Gimbal俯仰角（baijing_fuyang）扫描（0~0.0015度, 自动转成rad）结果已保存到 {csv_path_gimbal_fuyang}")

    csv_path_gimbal_fangwei = os.path.join(output_dir, "trackval_gimbal_fangwei.csv")
    gimbal_fangwei_degs = [round(x, 4) for x in np.arange(0.0, 0.0016, 0.0001)]
    gimbal_fangwei_rads = [x * math.pi * 1000 / 180 for x in gimbal_fangwei_degs]

    with open(csv_path_gimbal_fangwei, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["baijing_fangwei(deg)", "baijing_fangwei(rad)", "CCD_x", "CCD_y", "CCD_z", "Dist_from_origin"])
        for deg, mrad in zip(gimbal_fangwei_degs, gimbal_fangwei_rads):
            recv.set_angles(
                emit_direction=(0.0, 0.0, 1.0),
                baijing_fuyang=0.0,
                baijing_fangwei=mrad,
                FSM_fuyang=0.0,
                FSM_fangwei=0.0,
            )
            ccd_loc = recv.compute_location(chief_only=True, VISUAL=False)
            d = math.sqrt(
                (ccd_loc[0] - base_ccd[0]) ** 2 +
                (ccd_loc[1] - base_ccd[1]) ** 2 +
                (ccd_loc[2] - base_ccd[2]) ** 2
            )
            print(f"baijing_fangwei={deg:.4f} deg ({mrad:.8f} rad), CCD={ccd_loc}, Δ={d}")
            writer.writerow([deg, mrad] + list(ccd_loc) + [d])
    print(f"Gimbal方位角（baijing_fangwei）扫描（0~0.0015度, 自动转成rad）结果已保存到 {csv_path_gimbal_fangwei}")


    csv_path_fsm_fuyang = os.path.join(output_dir, "trackval_fsm_fuyang.csv")
    deg2mrad = math.pi / 180 * 1000  # 1 deg = 1000 * pi/180 mrad ≈ 17.4533
    fsm_fuyang_degs = [round(x, 4) for x in np.arange(0.0, 0.0151, 0.001)]  # 0, ..., 0.015
    fsm_fuyang_mrads = [x * deg2mrad for x in fsm_fuyang_degs]
    with open(csv_path_fsm_fuyang, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FSM_fuyang(deg)", "FSM_fuyang(mrad)", "CCD_x", "CCD_y", "CCD_z", "Dist_from_origin"])
        for deg, fuyang in zip(fsm_fuyang_degs, fsm_fuyang_mrads):
            recv.set_angles(
                emit_direction=(0.0, 0.0, 1.0),
                baijing_fuyang=0.0,
                baijing_fangwei=0.0,
                FSM_fuyang=fuyang,  # mrad
                FSM_fangwei=0.0,
            )
            ccd_loc = recv.compute_location(chief_only=True, VISUAL=False)
            d = math.sqrt(
                (ccd_loc[0] - base_ccd[0]) ** 2 +
                (ccd_loc[1] - base_ccd[1]) ** 2 +
                (ccd_loc[2] - base_ccd[2]) ** 2
            )
            print(f"FSM_fuyang={deg:.4f} deg ({fuyang:.8f} mrad), CCD={ccd_loc}, Δ={d}")
            writer.writerow([deg, fuyang] + list(ccd_loc) + [d])
    print(f"FSM俯仰角扫描（0~0.015度, 自动转成mrad）结果已保存到 {csv_path_fsm_fuyang}")

    # FSM偏转：方位（FSM_fangwei）从0到0.015，每步0.001
    csv_path_fsm_fangwei = os.path.join(output_dir, "trackval_fsm_fangwei.csv")
    fsm_fangwei_degs = [round(x, 4) for x in np.arange(0.0, 0.0151, 0.001)]
    fsm_fangwei_mrads = [x * deg2mrad for x in fsm_fangwei_degs]
    with open(csv_path_fsm_fangwei, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FSM_fangwei(deg)", "FSM_fangwei(mrad)", "CCD_x", "CCD_y", "CCD_z", "Dist_from_origin"])
        for deg, fangwei in zip(fsm_fangwei_degs, fsm_fangwei_mrads):
            recv.set_angles(
                emit_direction=(0.0, 0.0, 1.0),
                baijing_fuyang=0.0,
                baijing_fangwei=0.0,
                FSM_fuyang=0.0,
                FSM_fangwei=fangwei,  # mrad
            )
            ccd_loc = recv.compute_location(chief_only=True, VISUAL=False)
            d = math.sqrt(
                (ccd_loc[0] - base_ccd[0]) ** 2 +
                (ccd_loc[1] - base_ccd[1]) ** 2 +
                (ccd_loc[2] - base_ccd[2]) ** 2
            )
            print(f"FSM_fangwei={deg:.4f} deg ({fangwei:.8f} mrad), CCD={ccd_loc}, Δ={d}")
            writer.writerow([deg, fangwei] + list(ccd_loc) + [d])
    print(f"FSM方位角扫描（0~0.015度, 自动转成mrad）结果已保存到 {csv_path_fsm_fangwei}")

    csv_path_emit_direction = os.path.join(output_dir, "trackval_emit_direction.csv")
    deg_steps = [round(x, 7) for x in np.arange(0.0, 0.00151, 0.0001)]  # 单位deg，从0到0.0015deg，步长0.0001deg
    with open(csv_path_emit_direction, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["emit_direction_offset_deg", "emit_direction_x", "emit_direction_y", "emit_direction_z", "CCD_x", "CCD_y",
             "CCD_z", "Dist_from_origin"])
        for offset_deg in deg_steps:
            theta_rad = offset_deg * math.pi / 180  # 角度转弧度
            # 使与z正方向夹角为theta_rad，取x-z平面方向: (sinθ, 0, cosθ)
            emit_direction = (math.sin(theta_rad), 0.0, math.cos(theta_rad))
            # 归一化emit_direction（理论上应已归一，这里安全起见）
            norm = math.sqrt(emit_direction[0] ** 2 + emit_direction[1] ** 2 + emit_direction[2] ** 2)
            emit_direction_norm = (
                emit_direction[0] / norm,
                emit_direction[1] / norm,
                emit_direction[2] / norm
            )
            recv.set_angles(
                emit_direction=emit_direction_norm,
                baijing_fuyang=0.0,
                baijing_fangwei=0.0,
                FSM_fuyang=0.0,
                FSM_fangwei=0.0,
            )
            ccd_loc = recv.compute_location(chief_only=True, VISUAL=False)
            d = math.sqrt(
                (ccd_loc[0] - base_ccd[0]) ** 2 +
                (ccd_loc[1] - base_ccd[1]) ** 2 +
                (ccd_loc[2] - base_ccd[2]) ** 2
            )
            print(
                f"emit_direction_offset_deg={offset_deg:.7f} deg, emit_direction={emit_direction_norm}, CCD={ccd_loc}, Δ={d}")
            writer.writerow([offset_deg] + list(emit_direction_norm) + list(ccd_loc) + [d])
    print(f"emit_direction与z轴夹角扫描（0~0.0015 deg, 步进0.0001 deg）结果已保存到 {csv_path_emit_direction}")