import numpy as np
from math import pi
from roboticstoolbox import DHRobot, RevoluteMDH
from spatialmath import SE3
from FRA333_HW3_18_39 import endEffectorJacobianHW3, checkSingularityHW3, computeEffortHW3

# Define robot parameters
# These parameters represent the physical dimensions of the robot links and joints
d1, a2, a3, d4, d5, d6 = 0.0892, 0.425, 0.39243, 0.109, 0.093, 0.082

def print_matrix(matrix, title):
    """
    Function to print a matrix with a title.
    
    Args:
    matrix (numpy.ndarray): The matrix to be printed
    title (str): The title of the matrix
    
    This function formats the matrix for easy readability, aligning the values in columns.
    """
    print(f"\n{title}")
    print("-" * 40)
    for row in matrix:
        print(" ".join(f"{val:8.4f}" for val in row))

def analyze_robot():
    """
    Main function to analyze the robot's kinematics and dynamics.
    
    This function performs the following analyses:
    1. Jacobian matrix comparison
    2. Singularity check for various configurations
    3. Effort and end-effector force calculation
    
    It uses both custom-implemented functions and the Robotics Toolbox for comparison.
    """
    # Create robot model using DHRobot from Robotics Toolbox
    # This model defines the kinematic structure of the robot using Denavit-Hartenberg parameters
    robot = DHRobot([
        RevoluteMDH(alpha=0, a=0, d=d1, offset=pi),
        RevoluteMDH(alpha=pi/2, a=0, d=0, offset=0),
        RevoluteMDH(alpha=0, a=-a2, d=0, offset=0)
    ], name="3DOF_Robot", tool=SE3([
        [0, 0, -1, -(a3 + d6)],
        [0, 1, 0, -d5],
        [1, 0, 0, d4],
        [0, 0, 0, 1]
    ]))

    q_init = [0.0, 0.0, 0.0]  # Initial joint configuration (all joints at 0 position)
    w_init = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # Initial end-effector wrench (force and moment)

    print("\n" + "="*50)
    print("ตรวจคำตอบข้อ 1: เปรียบเทียบ Jacobian")
    print("="*50)
    
    # Calculate Jacobian using custom function and Robotics Toolbox
    J_custom = endEffectorJacobianHW3(q_init)
    J_rtb = robot.jacobe(q_init)
    
    print_matrix(J_custom, "Jacobian (คำนวณเอง):")
    print_matrix(J_rtb, "Jacobian (Robotics Toolbox):")

    print("\n" + "="*50)
    print("ตรวจคำตอบข้อ 2: ตรวจสอบ Singularity")
    print("="*50)
    
    # List of joint configurations to check for singularities
    q_list = [
        [0.0, -pi/4, -pi/4],
        [0.0, pi/4, pi/2],
        [0.0, 0.0, -pi/2],
        [-0.24866892, 0.22598268, -0.19647569],
        [-1.74, -1.06, -1.15],
        [0.0 , -pi/2 , -0.2]
    ]
    
    print(f"{'Configuration':<15} {'คำนวณเอง':<15} {'Robotics Toolbox':<20} {'Singularity Value (คำนวณเอง)':<30} {'Singularity Value (RTB)':<30}")
    print("-" * 110)
    
    for i, q in enumerate(q_list):
        # Check singularity using custom function
        flag_custom = checkSingularityHW3(q)
        J_custom = endEffectorJacobianHW3(q)
        J_custom_linear = J_custom[:3, :]  # Extract linear velocity components
        S_custom = abs(np.linalg.det(J_custom_linear))  # Calculate determinant

        # Check singularity using Robotics Toolbox
        J_rtb = robot.jacobe(q)
        J_rtb_linear = J_rtb[:3, :]  # Extract linear velocity components
        S_rtb = abs(np.linalg.det(J_rtb_linear))  # Calculate determinant
        flag_rtb = 1 if S_rtb < 0.001 else 0  # Threshold for singularity detection

        print(f"Config {i+1:<9} {flag_custom:<15} {flag_rtb:<20} {S_custom:<30.6f} {S_rtb:<30.6f}")

    print("\n" + "="*50)
    print("ตรวจคำตอบข้อ 3: คำนวณ Effort และแรงที่ปลายมือ")
    print("="*50)
    
    # Calculate effort using custom function and Robotics Toolbox
    effort_custom = computeEffortHW3(q_init, w_init)
    J = robot.jacobe(q_init)
    effort_rtb = robot.pay(W=w_init, J=J)  # Use pay() method for inverse dynamics

    print("Joint Efforts:")
    print(f"{'Joint':<10} {'คำนวณเอง':<15} {'Robotics Toolbox':<20}")
    print("-" * 45)
    for i in range(3):
        print(f"Joint {i+1:<5} {effort_custom[i]:15.4f} {effort_rtb[i]:20.4f}")

    def calc_forces(effort):
        """
        Calculate end-effector forces from joint efforts.
        
        Args:
        effort (list): Joint efforts
        
        Returns:
        list: Calculated end-effector forces
        
        This function uses the robot's geometry to convert joint efforts to end-effector forces.
        """
        return [
            effort[0] / (a2 + a3 + d6),  # Force in x-direction
            effort[1] / (a2 + a3 + d6),  # Force in y-direction
            effort[2] / (a3 + d6)        # Force in z-direction
        ]

    forces_custom = calc_forces(effort_custom)
    forces_rtb = calc_forces(effort_rtb)

    print("\nแรงที่ปลายมือ (N):")
    print(f"{'Direction':<10} {'คำนวณเอง':<15} {'Robotics Toolbox':<20}")
    print("-" * 45)
    for i, direction in enumerate(['f1', 'f2', 'f3']):
        print(f"{direction:<10} {forces_custom[i]:15.4f} {forces_rtb[i]:20.4f}")

if __name__ == "__main__":
    analyze_robot()