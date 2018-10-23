from lib import dataset as D
import pandas as pd

if __name__ == '__main__':
    train_ids = D.get_train_ids(drop_black=False, drop_vstrips=False, drop_empty=False, drop_few=False)
    images = D.read_train_images(train_ids)
    masks = D.read_train_masks(train_ids)
    depths = D.read_depths(train_ids)

    folds_by_salt = D.get_folds_vector('coverage', images, masks, depths, n_folds=5)
    folds_by_depth = D.get_folds_vector('depth', images, masks, depths, n_folds=5)
    folds_by_rnd = D.get_folds_vector(None, images, masks, depths, n_folds=5, random_state=42)

    pd.DataFrame.from_dict({'id': train_ids, 'fold': folds_by_salt}).to_csv('data/folds_by_salt.csv', index=False)
    pd.DataFrame.from_dict({'id': train_ids, 'fold': folds_by_depth}).to_csv('data/folds_by_depth.csv', index=False)
    pd.DataFrame.from_dict({'id': train_ids, 'fold': folds_by_rnd}).to_csv('data/folds_by_rnd_42.csv', index=False)

