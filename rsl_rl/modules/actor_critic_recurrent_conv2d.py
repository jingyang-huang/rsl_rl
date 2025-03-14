# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
   

class ActorCriticRecurrentConv2d(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        image_input_shape,  # (C, H, W) -> assumption that actor and critic get input of same size
        num_actions,
        # policy Cfg:
        conv_layers_params,
        conv_linear_output_size,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
    
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        actor_intermediate_observation_size = conv_linear_output_size + num_actor_obs
        critic_intermediate_observation_size = conv_linear_output_size + num_critic_obs
        
        self.image_obs_size = torch.prod(torch.tensor(image_input_shape)).item()

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        #* Define Convolutional
        self.actor_conv_2d = ConvolutionalNetwork(
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            conv_linear_output_size=conv_linear_output_size,
        )

        self.critic_conv_2d = ConvolutionalNetwork(
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            conv_linear_output_size=conv_linear_output_size,
        )

        print(f"Actor Conv: {self.actor_conv_2d}")
        print(f"Critic Conv: {self.critic_conv_2d}")

        #* Define RNN
        self.rnn_a = RNNLayer(actor_intermediate_observation_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.rnn_c = RNNLayer(critic_intermediate_observation_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.rnn_a}")
        print(f"Critic RNN: {self.rnn_c}")


    def reset(self, dones=None):
        self.rnn_a.reset(dones)
        self.rnn_c.reset(dones)


    def act(self, observations, masks=None, hidden_states=None):
        print(f"--act")
        proprio_obs = observations[:, : -self.image_obs_size]
        image_obs = observations[:, -self.image_obs_size :]
        print(f"act - observations: {observations.shape}")  # [512, 30055]
        print(f"act - proprio_obs: {proprio_obs.shape}")    # [512, 55]
        print(f"act - image_obs: {image_obs.shape}")        # [512, 30000]

        image_embedding = self.actor_conv_2d(image_obs)
        concat_obs = torch.cat((image_embedding, proprio_obs), dim=1)
        input_a = self.rnn_a(concat_obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))


    def act_inference(self, observations):
        print(f"--act_inference")
        proprio_obs = observations[:, : -self.image_obs_size]
        image_obs = observations[:, -self.image_obs_size :]

        image_embedding = self.actor_conv_2d(image_obs)
        concat_obs = torch.cat((image_embedding, proprio_obs), dim=1)
        input_a = self.rnn_a(concat_obs)
        return super().act_inference(input_a.squeeze(0))


    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        print(f"--evaluate")
        proprio_obs = critic_observations[:, : -self.image_obs_size]
        image_obs = critic_observations[:, -self.image_obs_size :]

        image_embedding = self.critic_conv_2d(image_obs)
        concat_critic_obs = torch.cat((image_embedding, proprio_obs), dim=1)
        input_c = self.rnn_c(concat_critic_obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))


    def get_hidden_states(self):
        return self.rnn_a.hidden_states, self.rnn_c.hidden_states



class RNNLayer(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            print(f"RNN-forward: out1 = {out.shape}")
            out = unpad_trajectories(out, masks)
            print(f"RNN-forward: out2 = {out.shape}")
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            hidden_state[..., dones == 1, :] = 0.0



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    


class ConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        image_input_shape,
        conv_layers_params,
        conv_linear_output_size,
    ):
        super().__init__()

        self.image_input_shape = image_input_shape  # (C, H, W)
        self.image_obs_size = torch.prod(torch.tensor(self.image_input_shape)).item()

        # Build conv network and get its output size
        self.conv_net = self.build_conv_net(conv_layers_params)
        with torch.no_grad():
            dummy_image = torch.zeros(1, *self.image_input_shape)
            conv_output = self.conv_net(dummy_image)            
            self.image_feature_size = conv_output.view(1, -1).shape[1]

        # Build the connection layers between conv net and mlp -> flattened into 1D vector
        self.conv_linear = nn.Linear(self.image_feature_size, conv_linear_output_size)
        self.layernorm = nn.LayerNorm(conv_linear_output_size)

        # Initialize the weights
        self._initialize_weights()

    def build_conv_net(self, conv_layers_params):
        layers = []
        in_channels = self.image_input_shape[0]
        for idx, params in enumerate(conv_layers_params[:-1]):
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    params["out_channels"],
                    kernel_size=params.get("kernel_size", 3),
                    stride=params.get("stride", 1),
                    padding=params.get("padding", 0),
                ),
                nn.BatchNorm2d(params["out_channels"]),
                nn.ReLU(inplace=True),
                ResidualBlock(params["out_channels"]) if idx > 0 else nn.Identity(),
            ])
            in_channels = params["out_channels"]
        last_params = conv_layers_params[-1]
        layers.append(
            nn.Conv2d(
                in_channels,
                last_params["out_channels"],
                kernel_size=last_params.get("kernel_size", 3),
                stride=last_params.get("stride", 1),
                padding=last_params.get("padding", 0),
            )
        )
        layers.append(nn.BatchNorm2d(last_params["out_channels"]))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.conv_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.conv_linear.weight, mode="fan_out", nonlinearity="tanh")
        nn.init.constant_(self.conv_linear.bias, 0)
        nn.init.constant_(self.layernorm.weight, 1.0)
        nn.init.constant_(self.layernorm.bias, 0.0)

    def forward(self, image_obs):
        print(f"forward: image_obs = {image_obs.shape}")
        batch_size = image_obs.size(0)
        image = image_obs.view(batch_size, *self.image_input_shape)
        print(f"forward: self.image_input_shape = {self.image_input_shape}")
        print(f"forward: image = {image.shape}")

        conv_features = self.conv_net(image)
        flattened_conv_features = conv_features.view(batch_size, -1)
        normalized_conv_output = self.layernorm(self.conv_linear(flattened_conv_features))
        print(f"forward: normalized_conv_output = {normalized_conv_output.shape}")
        return normalized_conv_output