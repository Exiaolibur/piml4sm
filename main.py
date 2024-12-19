### Main Python Script (main.py)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from model.pinn_model import PINN, physics_loss
from data_generator import generate_data


def main():

    
    with open("./cfg/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    L_d = config["motor_params"]["L_d"]
    L_q = config["motor_params"]["L_q"]
    R_s = config["motor_params"]["R_s"]
    psi_f = config["motor_params"]["psi_f"]





    # initializ
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=config["training_params"]["learning_rate"])


    inputs, targets = generate_data(config["data_params"]["num_samples"], L_d, L_q, psi_f)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)






    epochs = config["training_params"]["epochs"]
    batch_size = config["training_params"]["batch_size"]

    for epoch in range(epochs):
        for i in range(0, len(inputs), batch_size):
            x_batch = inputs[i:i + batch_size]
            y_batch = targets[i:i + batch_size]

            psi_d = x_batch[:, 0]
            psi_q = x_batch[:, 1]
            theta_m = x_batch[:, 2]

            i_d = y_batch[:, 0]
            i_q = y_batch[:, 1]
            tau_m = y_batch[:, 2]

            #############
            W_pred = model(x_batch)

            # compute loss
            loss = physics_loss(psi_d, psi_q, theta_m, i_d, i_q, tau_m, W_pred, L_d, L_q, psi_f)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    torch.save(model.state_dict(), "pinn_motor_model.pth")









if __name__ == '__main__':
    main()