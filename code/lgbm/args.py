import argparse

def str2bool(v): #args를 bool로 받을경우의 type
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--seed', default=42, type=int, help='')
    
    parser.add_argument('--data_dir', default='../../data', type=str, help='')
    parser.add_argument('--model_dir', default='models/', type=str, help='')
    parser.add_argument('--output_dir', default='outputs/', type=str, help='')
    
    parser.add_argument('--run', default='test', type=str, help='')
    parser.add_argument('--bagging', type=str2bool, default=True, help='train에서 bagging 사용 유무')
    parser.add_argument('--using_train', type=str2bool, default=True, help='train에서 test data(예측타깃 제외) 사용 유무')

    
    parser.add_argument('--n_iterations', default=200, type=int, help='')
    parser.add_argument('--lr', default=0.05, type=float, help='')
    
    parser.add_argument('--num_leaves', default=64, type=int, help='')
    parser.add_argument('--min_data_in_leaf', default=1, type=int, help='')
    parser.add_argument('--max_depth', default=16, type=int, help='')
    parser.add_argument('--early_stopping_round', default=3, type=int, help='')
    parser.add_argument('--max_bin', default=50, type=int, help='')
    
    parser.add_argument('--feats', default=[
                        #기본 정보
                        'KnowledgeTag', 'encoded_testId', 'encoded_testID1', 'encoded_testID2', 'encoded_testNum',

                        #시간 정보 
                        'normalized_elapsed','normalized_elapsed_user','normalized_elapsed_test', 'normalized_elapsed_user_test',
                        'hour', 'correct_per_hour', 'hour_mode',                        
                        'month','month_mean','week_num',

                        #누적 정보                                          
                        'past_user_count', 'past_user_correct', 'average_user_correct',

                        'past_assessmentItemID_count', 'past_assessmentItemID_correct', 'average_assessmentItemID_correct',
                        'past_testId_correct', 'past_testId_count', 'average_testId_correct',
                        'past_testID1_correct', 'past_testID1_count', 'average_testID1_correct',
                        'past_testID2_correct', 'past_testID2_count', 'average_testID2_correct',
                        'past_testNum_correct', 'past_testNum_count', 'average_testNum_correct',                       
                        'past_KnowledgeTag_count', 'past_KnowledgeTag_correct', 'average_KnowledgeTag_correct', 

                        'past_user_assessmentItemID_count', 'past_user_assessmentItemID_correct', 'average_user_assessmentItemID_correct',
                        'past_user_testId_correct', 'past_user_testId_count', 'average_user_testId_correct',
                        'past_user_testID1_correct', 'past_user_testID1_count', 'average_user_testID1_correct',
                        'past_user_testID2_correct', 'past_user_testID2_count', 'average_user_testID2_correct',
                        'past_user_testNum_correct', 'past_user_testNum_count', 'average_user_testNum_correct',                   
                        'past_user_KnowledgeTag_count', 'past_user_KnowledgeTag_correct', 'average_user_KnowledgeTag_correct',                        
                        
                        #전체 정보 
                        'test_sum', 'test_count', 'test_normalized_mean', 
                        'tag_sum', 'tag_count', 'tag_normalized_mean',
                        'item_sum', 'item_count', 'item_normalized_mean',                                                 
                        
                        #Shift 정보
                        'correct_ut_shift_1','correct_ut_shift_2','correct_ut_shift_3', 'correct_ut_shift_4', 'correct_ut_shift_5',
                        'recent_sum','recent_mean',
                        'average_assessmentItemID_correct_shift_1', 'average_assessmentItemID_correct_shift_2', 'average_assessmentItemID_correct_shift_3',
                        'normalized_elapsed_shift_1', 'normalized_elapsed_shift_2', 'normalized_elapsed_shift_3',
                        'normalized_elapsed_user_shift_1', 'normalized_elapsed_user_shift_2', 'normalized_elapsed_user_shift_3',
                        'normalized_elapsed_test_shift_1', 'normalized_elapsed_test_shift_2', 'normalized_elapsed_test_shift_3',
                        'normalized_elapsed_user_test_shift_1', 'normalized_elapsed_user_test_shift_2', 'normalized_elapsed_user_test_shift_3'
                        ], type=list, help='')
    
    args = parser.parse_args()
    return args
    