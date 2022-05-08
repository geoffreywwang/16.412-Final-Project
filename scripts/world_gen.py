from copy import deepcopy
import os
import math
import xml.etree.ElementTree as ET

def generate_gazebo_world(xs, objs, delta=0.5, output_filename='gen.world', template_filename='template.world', worlds_dir='../worlds'):
    # Create paths
    abs_worlds_dir = os.path.join(os.path.dirname(__file__), worlds_dir)

    # Parse template
    template = ET.parse(os.path.join(abs_worlds_dir, template_filename))
    
    # Create output XML
    root = ET.ElementTree(template.getroot())

    # Fill in human trajectory
    human = root.find("./world/actor[@name='human']/script/trajectory")
    temp_xs = [(obj.x, obj.y) for obj in objs][::1]
    human_poses = generate_poses(temp_xs, delta)
    for human_pose in human_poses:
        waypoint = ET.SubElement(human, 'waypoint')
        time = ET.SubElement(waypoint, 'time')
        pose = ET.SubElement(waypoint, 'pose')

        time.text = str(human_pose[0])
        pose.text = f'{human_pose[1]} {human_pose[2]} 0  0 0 {human_pose[3]}'

    # Generate robots
    robot_node = template.find("./world/actor[@name='robot']")
    for i in range(len(xs) - 1):
        root.find("./world").append(deepcopy(robot_node))

    # Find all robot nodes
    robots = root.findall("./world/actor[@name='robot']")

    # Modify robots
    for i, robot in enumerate(robots):

        # Set robot properties
        robot.set('name', f'robot{i}')

        # Set robot color
        if i == 1:
            material = ET.SubElement(robot.find('./link/visual'), 'material')
            ambient = ET.SubElement(material, 'ambient')
            ambient.text = f'0.0 0.8 0.0'
            diffuse = ET.SubElement(material, 'diffuse')
            diffuse.text = f'0.0 0.8 0.0'
        elif i == 2:
            material = ET.SubElement(robot.find('./link/visual'), 'material')
            ambient = ET.SubElement(material, 'ambient')
            ambient.text = f'0.8 0.0 0.0'
            diffuse = ET.SubElement(material, 'diffuse')
            diffuse.text = f'0.8 0.0 0.0'
        elif i == 3:
            material = ET.SubElement(robot.find('./link/visual'), 'material')
            ambient = ET.SubElement(material, 'ambient')
            ambient.text = f'0.0 0.0 0.8'
            diffuse = ET.SubElement(material, 'diffuse')
            diffuse.text = f'0.0 0.0 0.8'
        else:
            pass
        
        # Generate trajectory for robot
        traj = robot.find('./script/trajectory')
        robot_poses = generate_poses(xs[i], delta)
        for robot_pose in robot_poses:
            waypoint = ET.SubElement(traj, 'waypoint')
            time = ET.SubElement(waypoint, 'time')
            pose = ET.SubElement(waypoint, 'pose')

            time.text = str(robot_pose[0])
            pose.text = f'{robot_pose[1]} {robot_pose[2]} 0  0 0 {robot_pose[3]}'

    # Generate output path
    abs_output_path = os.path.join(abs_worlds_dir, output_filename)

    # Save generated world file
    root.write(abs_output_path)
    print(f'World file generated: {abs_output_path}')


def generate_poses(xs, delta):
    '''
    Generate poses from a list of coordinates
    '''

    i = 0
    poses = []
    while i < len(xs) - 1:
        if xs[i][0] - xs[i+1][0] != 0 or xs[i][1] - xs[i+1][1] != 0:
            poses.append((i*delta, xs[i][0], xs[i][1], math.atan2(xs[i+1][1] - xs[i][1], xs[i+1][0] - xs[i][0])))
        i += 1
    poses.append((i*delta, xs[i][0], xs[i][1], poses[-1][3]))

    return poses

if __name__ == "__main__":
    pass
