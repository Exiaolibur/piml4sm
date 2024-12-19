import torch
import torch.nn as nn





class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden1 = nn.Linear(3, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)  # 磁场能量 W
        self.activation = torch.tanh

    def forward(self, inputs):
        x = self.activation(self.hidden1(inputs))
        x = self.activation(self.hidden2(x))
        return self.output_layer(x)





##########################
# Physical loss Important！ for PIML constraint
#########################
def physics_loss(psi_d, psi_q, theta_m, i_d, i_q, tau_m, W_pred, L_d, L_q, psi_f):

    # stator current 
    i_d_pred = (psi_d - psi_f) / L_d
    i_q_pred = psi_q / L_q

    #torque
    tau_m_pred = 1.5 * (psi_d * i_q - psi_q * i_d)

    # loss
    current_loss = torch.mean((i_d_pred - i_d) ** 2 + (i_q_pred - i_q) ** 2)
    torque_loss = torch.mean((tau_m_pred - tau_m) ** 2)

    return current_loss + torque_loss