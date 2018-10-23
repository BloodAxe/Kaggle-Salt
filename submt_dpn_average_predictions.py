from tqdm import tqdm

from lib.ensemble import extract_oof_predictions, make_cv_submit

if __name__ == '__main__':
    # prefix = 'dpn_128_light'
    # inputs = [
    #     'experiments/dpn/128/light/dpn_128_light_zealous_pike_fold_salt_0',
    #     'experiments/dpn/128/light/dpn_128_light_jovial_darwin_fold_salt_1',
    #     'experiments/dpn/128/light/dpn_128_light_confident_bartik_fold_salt_2',
    #     'experiments/dpn/128/light/dpn_128_light_unruffled_minsky_fold_salt_3',
    #     'experiments/dpn/128/light/dpn_128_light_pedantic_wing_fold_salt_4',
    # ]


    # prefix = 'dpn_128pad_light'
    # inputs = [
    #     'experiments/dpn/128pad/light/dpn_128pad_light_cranky_stallman_fold_salt_0',
    #     'experiments/dpn/128pad/light/dpn_128pad_light_vibrant_tesla_fold_salt_1',
    #     'experiments/dpn/128pad/light/dpn_128pad_light_nervous_lichterman_fold_salt_2',
    #     'experiments/dpn/128pad/light/dpn_128pad_light_hopeful_shirley_fold_salt_3',
    #     'experiments/dpn/128pad/light/dpn_128pad_light_zen_meninsky_fold_salt_4',
    # ]

    prefix = 'ternaus_v3_128_light'
    inputs = [
        'ternaus_v3_128_light_loving_mayer_fold_salt_0_val_lb',
        # 'ternaus_v3_128_light_loving_mayer_fold_salt_0_val_lb_snapshot_0',
        # 'ternaus_v3_128_light_loving_mayer_fold_salt_0_val_lb_snapshot_1',
        # 'ternaus_v3_128_light_loving_mayer_fold_salt_0_val_lb_snapshot_2',
        # 'ternaus_v3_128_light_loving_mayer_fold_salt_0_val_lb_snapshot_3',
        # 'ternaus_v3_128_light_loving_mayer_fold_salt_0_val_lb_snapshot_4',

        'ternaus_v3_128_light_wizardly_poincare_fold_salt_1_val_lb',
        # 'ternaus_v3_128_light_wizardly_poincare_fold_salt_1_val_lb_snapshot_0',
        # 'ternaus_v3_128_light_wizardly_poincare_fold_salt_1_val_lb_snapshot_1',
        # 'ternaus_v3_128_light_wizardly_poincare_fold_salt_1_val_lb_snapshot_2',
        # 'ternaus_v3_128_light_wizardly_poincare_fold_salt_1_val_lb_snapshot_3',
        # 'ternaus_v3_128_light_wizardly_poincare_fold_salt_1_val_lb_snapshot_4',

        'ternaus_v3_128_light_blissful_hypatia_fold_salt_2_val_lb',
        # 'ternaus_v3_128_light_blissful_hypatia_fold_salt_2_val_lb_snapshot_0',
        # 'ternaus_v3_128_light_blissful_hypatia_fold_salt_2_val_lb_snapshot_1',
        # 'ternaus_v3_128_light_blissful_hypatia_fold_salt_2_val_lb_snapshot_2',
        # 'ternaus_v3_128_light_blissful_hypatia_fold_salt_2_val_lb_snapshot_3',
        # 'ternaus_v3_128_light_blissful_hypatia_fold_salt_2_val_lb_snapshot_4',

        'ternaus_v3_128_light_stoic_franklin_fold_salt_3_val_lb',
        # 'ternaus_v3_128_light_stoic_franklin_fold_salt_3_val_lb_snapshot_0',
        # 'ternaus_v3_128_light_stoic_franklin_fold_salt_3_val_lb_snapshot_1',
        # 'ternaus_v3_128_light_stoic_franklin_fold_salt_3_val_lb_snapshot_2',
        # 'ternaus_v3_128_light_stoic_franklin_fold_salt_3_val_lb_snapshot_3',
        # 'ternaus_v3_128_light_stoic_franklin_fold_salt_3_val_lb_snapshot_4',

        'ternaus_v3_128_light_epic_babbage_fold_salt_4_val_lb',
        # 'ternaus_v3_128_light_epic_babbage_fold_salt_4_val_lb_snapshot_0',
        # 'ternaus_v3_128_light_epic_babbage_fold_salt_4_val_lb_snapshot_1',
        # 'ternaus_v3_128_light_epic_babbage_fold_salt_4_val_lb_snapshot_2',
        # 'ternaus_v3_128_light_epic_babbage_fold_salt_4_val_lb_snapshot_3',
        # 'ternaus_v3_128_light_epic_babbage_fold_salt_4_val_lb_snapshot_4',
    ]

    for input in tqdm(inputs, desc='Extracting OOF', total=len(inputs)):
        extract_oof_predictions(input)

    make_cv_submit(inputs, prefix)
