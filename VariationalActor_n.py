import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import math
import torch
import torch.nn as nn
from ModularActor import ActorGraphPolicy
from TransformerActor import TransformerModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from utils import GLOBAL_SET_OF_NAMES, sinkhorn
from new_decoder_base import DecoderBase

from collections import OrderedDict, defaultdict
from transformers import BertTokenizer, BertModel

torch.set_printoptions(precision=None, threshold=1e10, edgeitems=None, linewidth=None, profile=None, sci_mode=None)


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
        # last_hidden_state = output.last_hidden_state
        last_hidden_state = output.last_hidden_state[:,0] # extract cls token embedding
        return input, last_hidden_state.squeeze(0)


class VariationalPolicy(ActorGraphPolicy):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_action,
            max_children,
            disable_fold,
            td,
            bu,
            envs_train,
            args=None,
    ):
        super(ActorGraphPolicy, self).__init__()
        self.args = args
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_size = args.variational_latent_size
        self.envs = envs_train

        self.actor = DecoderBase(
            frequency_encoding_size=args.variational_frequency_encoding_size,
            latent_size=args.variational_latent_size,
            d_model=args.variational_d_model,
            nhead=args.variational_nhead,
            obs_scale=args.variational_obs_scale,

            obs_z_in_init_w=args.variational_obs_z_in_init_w,
            act_z_in_init_w=args.variational_act_z_in_init_w,
            act_out_init_w=args.variational_act_out_init_w,

            num_transformer_blocks=args.variational_num_transformer_blocks,
            dim_feedforward=args.variational_dim_feedforward,

            dropout=args.variational_dropout,
            activation=args.variational_activation,
        ).to(device)

        self.obs_text_encodings = defaultdict(list) # list is also ordered
        self.action_encodings = defaultdict(OrderedDict)
        wrap = Bert_Model()

        for env_name, env in self.envs.items():
            action_dim = env.action_space.shape[0]
            # num_joints = len(env.unwrapped.data.dof_island)
            env_limb_info = env.env.model.body_names[1:]
            # we are assuming all limbs have these values in the same sequence(this is due to wrapped modular env)
            initial_obs_info = []
            limb_type0, limb_type1, limb_type2, limb_type3 = '0', '1', '2', '3'
            for limb in env_limb_info:
                if "torso" in limb:
                    other_lts = [limb_type0, 'not_' + limb_type1, 'not_' + limb_type2, 'not_' + limb_type3]
                elif ("thigh" or "1" or "knee") in limb:
                    other_lts = ['not_' + limb_type0, limb_type1, 'not_' + limb_type2, 'not_' + limb_type3]
                elif ("shin" or "leg" or "2" or "shoulder") in limb:
                    other_lts = ['not_' + limb_type0, 'not_' + limb_type1, limb_type2, 'not_' + limb_type3]
                elif ("foot" or "3" or "elbow") in limb:
                    # "0","1","2","3" will correspond to one hot value 1 while all "not {0,1,2,3}"
                    other_lts = ['not_' + limb_type0, 'not_' + limb_type1, 'not_' + limb_type2, limb_type3]
                    # will correspond to 0

                initial_obs_info.append([
                    limb + '_position_x', limb + '_position_y', limb + '_position_z',
                    limb + '_translational_velocity_x',
                    limb + '_translational_velocity_y', limb + '_translational_velocity_z',
                    limb + '_rotational_velocity_x',
                    limb + '_rotational_velocity_y', limb + '_rotational_velocity_z', limb + '_exponential_map_x',
                    limb + '_exponential_map_y', limb + '_exponential_map_z', other_lts[0], other_lts[1], other_lts[2],
                    other_lts[3],
                    limb + '_angle', limb + '_joint_range_x', limb + '_joint_range_y'])

            obs_inf = []
            for i in range(len(initial_obs_info)):
                for j in range(len(initial_obs_info[i])):
                    obs_inf.append(initial_obs_info[i][j])
            for joint_info in obs_inf:
                _, obs_text_encoding = wrap.encode_info(joint_info)
                self.obs_text_encodings[env_name].append(obs_text_encoding)
                # self.obs_text_encodings[env_name].append(torch.mean(obs_text_encoding, 0))

            action_info = [motor + '_ctrl' for motor in env.motors]
            action_info.insert(0, 'torso_ctrl')
            act_inf = OrderedDict()
            for i in range(action_dim + 1):
                act_inf[action_info[i]] = 0
            for actuator, value in act_inf.items():
                if actuator not in self.action_encodings[env_name].keys():
                    _, act_encoding = wrap.encode_info(actuator)
                    self.action_encodings[env_name][actuator] = act_encoding
                    # self.action_encodings[env_name][actuator] = torch.mean(act_encoding, 0)

    def forward(self, state, env_name, mode="train"):
        self.clear_buffer()

        if mode == "inference":
            temp, self.batch_size = self.batch_size, 1

        obs_encodings = torch.stack(self.obs_text_encodings[env_name]).unsqueeze(0)

        obs_encodings = obs_encodings.expand(self.batch_size,
                             obs_encodings.shape[1], obs_encodings.shape[2])
        act_encodings = torch.stack(list(self.action_encodings[env_name].values()), dim=0).unsqueeze(0)  # (1,7,768)
        act_encodings =  act_encodings.expand(self.batch_size,
            act_encodings.shape[1], act_encodings.shape[2])
        self.action = self.actor(obs_encodings, act_encodings, state.permute(1, 0))
        self.action = self.max_action * torch.tanh(self.action)

        if mode == "inference":
            self.batch_size = temp

        return self.action

    def change_morphology(self, parents, action_ids):

        self.parents = parents
        self.action_ids = torch.LongTensor(action_ids).to(device)
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs


class VariationalPolicy2(ActorGraphPolicy):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_action,
            max_children,
            disable_fold,
            td,
            bu,
            args=None,
    ):
        super(ActorGraphPolicy, self).__init__()
        self.args = args
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = TransformerModel(
            self.state_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm).to(device)

        self.perm_left_embedding = nn.Embedding(
            self.state_dim *
            len(GLOBAL_SET_OF_NAMES),
            args.attention_embedding_size).to(device)

        self.perm_right_embedding = nn.Embedding(
            self.state_dim *
            len(GLOBAL_SET_OF_NAMES),
            args.attention_embedding_size).to(device)

        self.permutation_weight = nn.Parameter(torch.empty((
            len(GLOBAL_SET_OF_NAMES),
            args.attention_embedding_size,
            args.attention_embedding_size)).to(device))

        self.permutation_weight.data.uniform_(-0.01, 0.01)

    def get_amorpheus_perm_slice(self):

        perm_ids = torch.arange(self.num_limbs * self.state_dim,
                                dtype=torch.int64, device=device)

        perm_ids = torch.stack([perm_ids.roll(
            -self.state_dim * limb_id) for limb_id in range(self.num_limbs)])

        perm = nn.functional.one_hot(
            perm_ids, self.state_dim * self.num_limbs).to(torch.float32)

        return perm.view(
            self.num_limbs, 1,
            self.state_dim * self.num_limbs,
            self.state_dim * self.num_limbs)

    def get_learned_perm_slice(self):

        obs_ids = torch.repeat_interleave(
            self.action_ids, self.state_dim) * self.state_dim + torch.arange(
            self.state_dim,
            dtype=torch.int64, device=device).repeat(self.num_limbs)

        perm_left_embeddings = self.perm_left_embedding(obs_ids).view(
            1, 1, self.state_dim * self.num_limbs,
            self.args.attention_embedding_size).contiguous()

        perm_right_embeddings = self.perm_right_embedding(obs_ids).view(
            1, 1, self.state_dim * self.num_limbs,
            self.args.attention_embedding_size).contiguous()

        perm_weights = nn.functional.embedding(
            self.action_ids,
            self.permutation_weight.view(len(GLOBAL_SET_OF_NAMES), -1)).view(
            self.num_limbs, 1,
            self.args.attention_embedding_size,
            self.args.attention_embedding_size).contiguous()

        return sinkhorn(perm_left_embeddings @
                        perm_weights @
                        perm_right_embeddings.permute(0, 1, 3, 2)).exp()

    def forward(self, state, mode="train"):

        self.clear_buffer()

        if mode == "inference":
            temp, self.batch_size = self.batch_size, 1

        state = state.view(1, self.batch_size, self.state_dim * self.num_limbs, 1)

        perm = self.get_learned_perm_slice()
        self.perm_loss = ((perm - self.get_amorpheus_perm_slice()) ** 2).mean()

        self.input_state = torch.matmul(perm, state)[:, :, :self.state_dim, 0]

        self.action = self.actor(self.input_state)
        self.action = self.max_action * torch.tanh(self.action)

        # because of the permutation of the states, we need to
        # unpermute the actions now so that the actions are (batch,actions)
        self.action = self.action.permute(1, 0, 2)

        if mode == "inference":
            self.batch_size = temp

        return torch.squeeze(self.action)

    def change_morphology(self, parents, action_ids):
        self.parents = parents
        self.action_ids = torch.LongTensor(action_ids).to(device)
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
