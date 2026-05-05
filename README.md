### Training

To train on a specific morphology mixture, for example, walker_humanoids_hopper:

```
python3 $HOME/AnyMorph/modular-rl/src/main_n.py --custom_xml environments/walker_humanoids_hopper --label walker_humanoids_hopper-withoutltv-universal --expID walker_humanoids_hopper-withoutltv-universal-0 --seed 0 --lr 0.00005 --grad_clipping_value 0.1 --actor_type variational --critic_type transformer --attention_layers 3
--attention_heads 2 --attention_hidden_size 256 --transformer_norm 1 --condition_decoder 1 --variational_frequency_encoding_size 96 --variational_latent_size 768 --variational_d_model 128 --variational_nhead 2 --variational_obs_scale 1000.0 --variational_act_scale 1000.0 --variational_obs_z_in_init_w 0.0
--variational_act_z_in_init_w 0.0 --variational_act_out_init_w 0.003 --variational_num_transformer_blocks 3 --variational_dim_feedforward 256 --variational_dropout 0.0 --variational_activation relu &
```

### Generalization

```
 python3 plot_generalization.py --expID walker_humanoids_hopper-withoutltv-universal-0 --custom_xml environments/walker_humanoids_hopper --custom_xml_held_out environments/walker_humanoids_hopper_test --seed 0 --lr 0.00005 --grad_clipping_value 0.1 --actor_type variational --critic_type transformer --attention_layers 3
 --attention_heads 2 --attention_hidden_size 256 --transformer_norm 1 --condition_decoder 1 --variational_frequency_encoding_size 96 --variational_latent_size 768 --variational_d_model 128 --variational_nhead 2 --variational_obs_scale 1000.0 --variational_act_scale 1000.0 --variational_obs_z_in_init_w 0.0
--variational_act_z_in_init_w 0.0 --variational_act_out_init_w 0.003 --variational_num_transformer_blocks 3 --variational_dim_feedforward 256 --variational_dropout 0.0 --variational_activation relu
```

### Experiments and Graphs  

The comprehensive experiments,results and insights are all available in this report.
