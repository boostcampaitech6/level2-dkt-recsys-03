import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    
    # -- 데이터 경로 및 파일 이름 설정
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="outputs/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    # -- 최대 시퀀스 길이 설정
    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    
    # GPU 설정
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")
    

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstm", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )

    # K-FOLD 사용 여부 결정
    parser.add_argument("--split_method", default="general", type=str, help="data split strategy -- general OR k-fold")
    parser.add_argument("--n_splits", default=5, type=int, help="number of k-fold splits")
    
    # WandB 설정
    parser.add_argument("--wandb_project_name", default="dkt", type=str, help="Setting WandB Project Name")
        
    ### Tfixup 관련 ###
    parser.add_argument("--Tfixup", default=False, type=bool, help="Tfixup")

    # LQTR argument
    parser.add_argument("--out_dim", default=128, type=int, help="LQTR linear hidden dim")

    #augmentation
    parser.add_argument("--aug", default=False, type=bool, help="augmentation")
    parser.add_argument("--window", default=20, type=int, help="window length")
    
    # Argumentation 관련 #
    parser.add_argument("--window", default=True, type=bool, help="Arumentation with stride window")
    parser.add_argument("--shuffle", default=False, type=bool, help="data shuffle option")
    parser.add_argument("--stride", default=101, type=int)
    parser.add_argument("--shuffle_n", default= 3, type=int)

    args = parser.parse_args()

    return args
