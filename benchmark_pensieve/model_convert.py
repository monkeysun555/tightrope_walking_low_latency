import tensorflow.saved_model as sm
import static_a3c as a3c
import tensorflow as tf
import os

S_INFO = 8
S_LEN = 12
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 1
STARTING_EPOCH = 60000

NN_MODEL = 'nn_model_s_2_ep_' + str(STARTING_EPOCH) + '.ckpt'
SM_MODEL = 'sm_2s_ep_' + str(STARTING_EPOCH)

def get_model_saving_location():
	return os.path.join(os.path.dirname(os.path.relpath(__file__)),'./models/', SM_MODEL)


def main():
	with tf.Session() as sess:
		actor = a3c.ActorNetwork(sess,
									state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
									learning_rate=ACTOR_LR_RATE)
		critic = a3c.CriticNetwork(sess,
									state_dim=[S_INFO, S_LEN],
									learning_rate=CRITIC_LR_RATE)

		sess.run(tf.global_variables_initializer())
		ckpt_saver = tf.train.Saver()  # save neural net parameters


		nn_model = os.path.join(os.path.dirname(os.path.relpath(__file__)), './models/', NN_MODEL)
		ckpt_saver.restore(sess, nn_model)

		model_saving_location = get_model_saving_location()
		sm_saver = sm.builder.SavedModelBuilder(model_saving_location)

		model_input = {'inputs': sm.utils.build_tensor_info(actor.inputs)}
		model_output = {'outputs': sm.utils.build_tensor_info(actor.out)}
		model_infer_signature = sm.signature_def_utils.build_signature_def(model_input, model_output,
																		   'getBr_idx')
		# sm_saver.add_meta_graph_and_variables(sess, ['actor_model'], {'sig_getBr_idx': model_infer_signature})
		
		sm_saver.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], 
					{'sig_getBr_idx': model_infer_signature}, 
					strip_default_attrs=True)

		print [str(n.name) for n in tf.get_default_graph().as_graph_def().node]
		# print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph())) 
		# print(sess.graph.get_operations())
		sm_saver.save()

if __name__ == '__main__':
	main()