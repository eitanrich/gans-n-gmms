import tensorflow as tf
import os
import sys
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import mfa
import mfa_tf
import mfa_utils
import image_batch_provider


def split_data_by_model_components(gmm_model, model_folder, image_provider, image_shape, batch_size=200, whiten=False):
    out_folder = os.path.join(model_folder, 'component_lists')

    if whiten:
       dataset_mean, dataset_std = mfa_utils.get_dataset_mean_and_std(image_provider)
    else:
        dataset_mean, dataset_std = (0.0, 1.0)

    # Restore the TF model
    G_PI, G_MU, G_A, G_D = mfa_tf.init_raw_parms_from_gmm(gmm_model)
    Theta_G = (G_PI, G_MU, G_A, G_D)
    X = tf.placeholder(tf.float32, shape=[None, gmm_model.components[0]['A'].shape[0]])
    C_X = mfa_tf.get_max_posterior_component(X, *Theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    comp_list = defaultdict(list)

    def collect_components(image_list, group_num):
        num_images = len(image_list)
        for idx in range(0, num_images, batch_size):
            idx_end = min(idx+batch_size, num_images)
            print('Processing images ', idx, 'to', idx_end, 'of', num_images)

            # Test the original batch and its mirror version
            orig_batch = image_provider.get_images_from_list(image_list[idx:idx_end])
            # Whiten the data
            orig_batch = (orig_batch - dataset_mean) / dataset_std
            m = orig_batch.shape[0]
            w, h = image_shape
            batch_ml_comps = sess.run(C_X, feed_dict={X: orig_batch})
            mirror_batch = orig_batch.reshape([m, h, w, 3])[:, :, ::-1, :].reshape([m, -1])
            mirror_ml_comp = sess.run(C_X, feed_dict={X: mirror_batch})

            for j in range(m):
                image_line = '%s %d' % (image_list[idx+j], group_num)
                comp_list[batch_ml_comps[j]].append(image_line)
                comp_list[mirror_ml_comp[j]].append('mirror:'+image_line)

    print('Collecting train set MAP components...')
    collect_components(image_provider.train_image_list, 0)

    print('Collecting test set MAP components...')
    collect_components(image_provider.test_image_list, 1)

    print('Writing results...')
    os.makedirs(out_folder, exist_ok=True)
    for c_num, c_list in comp_list.items():
        with open(os.path.join(out_folder, 'comp_{}.txt'.format(c_num)), 'w') as out_file:
            for item in c_list:
                out_file.write("%s\n" % item)


def flatten_hierarchic_model(root_gmm, model_folder):
    num_comps = len(root_gmm.components)
    all_comps = {}
    for i in range(num_comps):
        comp_gmm = mfa.MFA()
        comp_folder = os.path.join(model_folder, 'hierarchic_model', 'comp_{}'.format(i))
        comp_gmm.load(os.path.join(comp_folder, 'final_gmm'))
        num_sub_comps = len(comp_gmm.components)
        for j in range(num_sub_comps):
            comp_num = len(all_comps)
            all_comps[comp_num] = comp_gmm.components[j]
            all_comps[comp_num]['pi'] *= root_gmm.components[i]['pi']
            print('Component', i, '/', j, 'pi=', all_comps[comp_num]['pi'])
    flat_gmm = mfa.MFA(all_comps)
    total_pi = sum([c['pi'] for c in flat_gmm.components.values()])
    assert abs(total_pi-1.0) < 1e-5
    flat_gmm.components[0]['pi'] = 1.0 - (total_pi - flat_gmm.components[0]['pi'])
    flat_gmm.save(os.path.join(model_folder, 'final_flat_model'))
    print('Total number of components:', len(flat_gmm.components))
