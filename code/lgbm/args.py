import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--seed', default=42, type=int, help='')
    
    parser.add_argument('--data_dir', default='../../data', type=str, help='')
    parser.add_argument('--model_dir', default='models/', type=str, help='')
    parser.add_argument('--output_dir', default='outputs/', type=str, help='')
    
    parser.add_argument('--run', default='test', type=str, help='')
    
    parser.add_argument('--n_iterations', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    
    parser.add_argument('--num_leaves', default=64, type=int, help='') #256
    parser.add_argument('--min_data_in_leaf', default=3, type=int, help='')
    parser.add_argument('--max_depth', default=32, type=int, help='') #6
    parser.add_argument('--early_stopping_round', default=3, type=int, help='')
    parser.add_argument('--max_bin', default=100, type=int, help='')
    
    parser.add_argument('--feats', default=['KnowledgeTag', 'encoded_testId', 'encoded_testID1', 'encoded_testID2', 'encoded_testNum',
                        #누적 정보
                                          
                        #'user_correct_answer', 'user_total_answer', 'user_acc',                           
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
                        
                        'average_assessmentItemID_correct_shift_1', 'average_assessmentItemID_correct_shift_2', 'average_assessmentItemID_correct_shift_3',
                        #전체 정보 
                        'test_sum', 'test_count', 'test_normalized_mean',# 'test_mean', 
                        'tag_sum', 'tag_count', 'tag_normalized_mean', #'tag_mean', 
                        'item_sum', 'item_count', 'item_normalized_mean', #'item_mean', 
                        #과거 정보
                        #'correct_shift_1', 'correct_shift_2', 'correct_shift_3', 'correct_shift_4', 'correct_shift_5',
                        'recent_sum','recent_mean',
                        'correct_ut_shift_1','correct_ut_shift_2','correct_ut_shift_3', 'correct_ut_shift_4', 'correct_ut_shift_5',
                        #시간
                        'elapsed', 'elapsed_median',
                        'normalized_elapsed','normalized_elapsed_user','normalized_elapsed_test', 'noramalized_elapsed_user_test',
                        'normalized_elapsed_shift_1', 'normalized_elapsed_shift_2', 'normalized_elapsed_shift_3',
                        'normalized_elapsed_user_shift_1', 'normalized_elapsed_user_shift_2', 'normalized_elapsed_user_shift_3',
                        'normalized_elapsed_test_shift_1', 'normalized_elapsed_test_shift_2', 'normalized_elapsed_test_shift_3',
                        'normalized_elapsed_user_test_shift_1', 'normalized_elapsed_user_test_shift_2', 'normalized_elapsed_user_test_shift_3',
                        'hour', 'correct_per_hour', 'hour_mode',                        
                        'month','month_mean','week_num'
                        
                        ], type=list, help='')
    
    args = parser.parse_args()
    return args
    