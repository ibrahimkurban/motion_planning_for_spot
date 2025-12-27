from pydrake.all import PackageMap, RigidTransform
import os

def ConfigureParser(parser):
    """Add the underactuated module packages to the given Parser."""
    package_xml = os.path.join(os.path.dirname(__file__), "../models/package.xml")
    parser.package_map().AddPackageXml(filename=package_xml)
    # Add spot_description
    parser.package_map().AddRemote(
        package_name="spot_description",
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/bdaiinstitute/spot_ros2/archive/097f46bd7c5b1d6bf0189a895c28ae0a90d287b1.tar.gz"
            ],
            sha256=("59df2f0644bd7d937365e580a6d3b69822e7fb0358dc58456cd0408830641d88"),
            strip_prefix="spot_ros2-097f46bd7c5b1d6bf0189a895c28ae0a90d287b1/spot_description/",
        ),
    )

def load_urdf_file(file_path):
    with open(file_path, 'r') as file:
        urdf_string = file.read()
    return urdf_string


# generate rectangle boxes for stepping stones
def generate_urdf_string(name, length, width, height, x, y):
    urdf = f"""<?xml version='1.0' encoding='us-ascii'?>
        <robot name={name}>
            <link name={name}>
                <visual>
                    <origin xyz="{x} {y} {height/2}" rpy="0 0 0" />
                    <geometry>
                        <box size="{length} {width} {height}" />
                    </geometry>
                    <material name="brown">
                        <color rgba="0.631 0.424 0.157 1" />
                    </material>
                </visual>
                <collision>
                    <origin xyz="{x} {y} {height/2}" rpy="0 0 0" />
                    <geometry>
                        <box size="{length} {width} {height}" />
                    </geometry> 
                    <surface>
                        <friction>
                        <ode>
                            <mu>1.0</mu>  <!-- Coefficient of friction -->
                            <mu2>1.0</mu2> <!-- Optional anisotropic friction coefficient -->
                        </ode>
                        </friction>
                    </surface>
                </collision>
                <inertial>
                    <mass value="1.8330189874700853" />
                    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1" />
                    <origin xyz="{x} {y} {height/2}" rpy="0 0 0" />
                </inertial>
            </link>
        </robot>
    """
    return urdf

# to generate stones and wel everything together:
def add_boxes_urdf_from_terrain(terrain, parser, plant, height=0.1):
    for stone in terrain.stepping_stones:
        x = stone.center[0]
        y = stone.center[1]
        length = stone.width
        width = stone.height

        stone_urdf = generate_urdf_string(f'"{stone.name}"', length, width, height, x, y)
        # print(stone_urdf)
        parser.AddModelsFromString(stone_urdf, "urdf")

        plant.WeldFrames(
            plant.world_frame(), # Attach to world frame
            plant.GetFrameByName(f'{stone.name}'), # Attach to box grame
            RigidTransform([0,0,0]) # Weld location
        )