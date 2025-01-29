from __future__ import print_function
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from TransformerActor import TransformerModel
from utils import GLOBAL_SET_OF_NAMES, sinkhorn
from new_decoder_base import DecoderBase
from collections import OrderedDict, defaultdict
from transformers import BertTokenizer, BertModel

class Bert_Model():
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = 'right'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.bert_model.eval()
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def encode_info(self, info):
        input = self.tokenizer(info, return_tensors="pt").to(device)
        output = self.bert_model(**input)
        last_hidden_state = output.last_hidden_state
        return input, last_hidden_state.squeeze(0)

class CriticVariationalPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_children,
            disable_fold,
            td,
            bu,
        envs_train,
            args=None,
    ):
        super().__init__()
        self.num_limbs = 1
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_size = args.variational_latent_size
        self.envs = envs_train

        self.critic1 = DecoderBase(
            frequency_encoding_size=args.variational_frequency_encoding_size,
            latent_size=args.variational_latent_size,
            d_model=args.variational_d_model,
            nhead=args.variational_nhead,
            obs_scale=args.variational_obs_scale,
            act_scale=args.variational_act_scale,

            is_critic=True,

            obs_z_in_init_w=args.variational_obs_z_in_init_w,
            act_z_in_init_w=args.variational_act_z_in_init_w,
            act_out_init_w=args.variational_act_out_init_w,

            num_transformer_blocks=args.variational_num_transformer_blocks,
            dim_feedforward=args.variational_dim_feedforward,

            dropout=args.variational_dropout,
            activation=args.variational_activation,
        ).to(device)

        self.critic2 = DecoderBase(
            frequency_encoding_size=args.variational_frequency_encoding_size,
            latent_size=args.variational_latent_size,
            d_model=args.variational_d_model,
            nhead=args.variational_nhead,
            obs_scale=args.variational_obs_scale,
            act_scale=args.variational_act_scale,

            is_critic=True,

            obs_z_in_init_w=args.variational_obs_z_in_init_w,
            act_z_in_init_w=args.variational_act_z_in_init_w,
            act_out_init_w=args.variational_act_out_init_w,

            num_transformer_blocks=args.variational_num_transformer_blocks,
            dim_feedforward=args.variational_dim_feedforward,

            dropout=args.variational_dropout,
            activation=args.variational_activation,
        ).to(device)

        # self.obs_embeddings = nn.Embedding(
        #     self.state_dim * len(GLOBAL_SET_OF_NAMES), self.latent_size).to(device)
        #
        # self.act_embeddings = nn.Embedding(
        #     len(GLOBAL_SET_OF_NAMES), self.latent_size).to(device)

        # ******************************************#
        self.obs_text_encodings = defaultdict(OrderedDict)
        self.action_encodings = defaultdict(OrderedDict)
        wrap = Bert_Model()

        # Todo: this will work well only if the names of joints and actuators are different across xmls
        for env_name, env in self.envs.items():
            action_dim = env.action_space.shape[0]
            # num_joints = len(env.unwrapped.data.dof_island)
            env_limb_info = env.env.model.body_names[1:]
            # we are assuming all limbs have these values in the same sequence(this is due to wrapped modular env)
            initial_obs_info = [
                [limb + '_position_x', limb + '_position_y', limb + '_position_z', limb + '_translational_velocity_x',
                 limb + '_translational_velocity_y', limb + '_translational_velocity_z',
                 limb + '_rotational_velocity_x',
                 limb + '_rotational_velocity_y', limb + '_rotational_velocity_z', limb + '_exponential_map_x',
                 limb + '_exponential_map_y', limb + '_exponential_map_z', limb + '_angle', limb + '_joint_range_x',
                 limb + '_joint_range_y'
                 ] for limb in env_limb_info]

            action_info = [motor + '_ctrl' for motor in env.env.model.actuator_names]
            action_info.insert(0, 'torso_ctrl')

            obs_inf = OrderedDict()
            for i in range(len(initial_obs_info)):
                for j in range(len(initial_obs_info[i])):
                    obs_inf[initial_obs_info[i][j]] = 0
            # change the code for both obs and act encodings such that self.obs[env_name][joint_name]=__
            for joint_info, value in obs_inf.items():
                if joint_info not in self.obs_text_encodings[env_name].keys():
                    _, obs_text_encoding = wrap.encode_info(joint_info)
                    self.obs_text_encodings[env_name][joint_info] = torch.mean(obs_text_encoding, 0)

            act_inf = OrderedDict()
            for i in range(action_dim + 1):
                act_inf[action_info[i]] = 0
            for actuator, value in act_inf.items():
                if actuator not in self.action_encodings[env_name].keys():
                    _, act_encoding = wrap.encode_info(actuator)
                    self.action_encodings[env_name][actuator] = torch.mean(act_encoding, 0)

    def forward(self, state, action, env_name):
        self.clear_buffer()

        obs_encodings = torch.stack(list(self.obs_text_encodings[env_name].values())).unsqueeze(0)
        obs_encodings = obs_encodings.expand(self.batch_size,
                             obs_encodings.shape[1], obs_encodings.shape[2])
        act_encodings = torch.stack(list(self.action_encodings[env_name].values()), dim=0).unsqueeze(0)  # (1,7,768)
        act_encodings =  act_encodings.expand(self.batch_size,
            act_encodings.shape[1], act_encodings.shape[2])

        self.x1 = self.critic1(obs_encodings, act_encodings, state.permute(1, 0),
                               act=action.permute(1, 0))

        self.x2 = self.critic2(obs_encodings, act_encodings, state.permute(1, 0),
                               act=action.permute(1, 0))

        return self.x1, self.x2

    def Q1(self, state, action, env_name):
        self.clear_buffer()
        obs_encodings = torch.stack(list(self.obs_text_encodings[env_name].values())).unsqueeze(0)
        obs_encodings = obs_encodings.expand(self.batch_size,
                             obs_encodings.shape[1], obs_encodings.shape[2])
        act_encodings = torch.stack(list(self.action_encodings[env_name].values()), dim=0).unsqueeze(0)  # (1,7,768)
        act_encodings =  act_encodings.expand(self.batch_size,
            act_encodings.shape[1], act_encodings.shape[2])

        self.x1 = self.critic1(obs_encodings, act_encodings,
                               state.permute(1, 0),
                               act=action.permute(1, 0))

        return self.x1

    def clear_buffer(self):
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents, action_ids):
        self.parents = parents
        self.action_ids = torch.LongTensor(action_ids).to(device)
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
