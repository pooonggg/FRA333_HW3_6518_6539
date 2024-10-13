# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส(ธนวัฒน์_6461)
1. ฐิติรัตน์_6518
2. พงศกร_6539
'''
#=============================================<คำตอบข้อ 1>======================================================#
#code here
from HW3_utils import FKHW3
import numpy as np
from math import pi

q_i = np.array([0.0 , 0.0 , 0.0])
q_singular = np.array([0.0 , -pi/2 , -0.2])
epsilon = 1e-3
w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #(Fx, Fy, Fz, Tx, Ty, Tz)

def endEffectorJacobianHW3(q:list[float])->list[float]:
    R,P,R_e,p_e = FKHW3(q)
    n_joints = len(q)
    J = np.zeros((6, n_joints))
    
    for i in range(n_joints):
        p_i = P[:,i]
        z = R[:,2,i] # เอาแค่ Rotation ของ Z มาใช้
        J[:3,i] = (np.cross(z,(p_e - p_i))) @ R_e # linear velocity
        J[3:,i] = z @ R_e# angular velocity
        
    return J

#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:
    # คำนวณ Jacobian
    J = endEffectorJacobianHW3(q)
    
    # ใช้เฉพาะส่วนของ Jacobian ที่เป็น 3x3 (เชิงเส้น)
    J_reduced = J[:3, :]  # ลดรูป Jacobian สำหรับตรวจหา Singularity

    # คำนวณ determinant ของ Jacobian ที่ลดรูปแล้ว
    det_J = np.linalg.det(J_reduced)
    # คำนวณหาค่า norm จาก det_J
    Norm_J = np.linalg.norm(det_J)

    # ใช้ค่า Norm เพื่อตรวจสอบ Singularity
    if  Norm_J < epsilon:
        return 1  # Singularity
    else:
        return 0  # ปกติ

    
#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    # Jacobian ของหุ่นยนต์
    J = endEffectorJacobianHW3(q)
    # คำนวณ Jacobian Transpose
    J_T = np.transpose(J)
    # คำนวณค่า torque สำหรับข้อต่อ
    tau = J_T @ w  
    return tau
#==============================================================================================================#
# endEffectorJacobianHW3(q_i)
# flag = checkSingularityHW3(q_singular)
# tau = computeEffortHW3(q_i,w)
# print(tau)