## Usage


Weight search (Phase 1) — `--weight_search directional` (default): the model is first
warmed up with uniform band weights (anchor `w = 1`), then each band's log-weight
is probed independently at `exp(±h)` and shifted toward the direction that raises
`pAUROC + pAUPRC` (coordinate-wise finite-difference ascent in log space).
`--weight_search random` keeps the original `Uniform(0.1, 3.0)` grid search for ablation.
## Requirements
```
- Python 3.9
- PyTorch 2.0.0+cu118
- Tqdm 4.64.1
- Scikit-learn 1.3.2
- Scipy 1.9.1

```
## Command line
python run.py --dataset Reddit --lr 1e-3 --weight_decay 1e-5 --alpha 0.5 --beta 0.8 --lambda_aff 3.0 --lambda_dev 0.1 --lambda_dgi 0.1 --lambda_rec 0.1 --lambda_cross 0.5 --lambda_score 0.1 --lambda_agree 0.2 --w_dgi 1.0 --w_rec 0.1 --wavelet_shape 1.0 --num_filters 2  --runs 10 --device cuda --gpu 2  --lambda_decorr 0.5

python run.py --dataset Facebook --lr 1e-3  --weight_decay 1e-3  --alpha 1.0 --beta 0.1 --lambda_aff 0.4 --lambda_dev 0.1 --lambda_dgi 0.5 --lambda_rec 0.1 --lambda_cross 0.5 --lambda_score 0.4 --lambda_agree 1.0 --w_dgi 3.0 --w_rec 0.1   --wavelet_shape 1.0  --num_filters 4  --runs 10 --device cuda --gpu 2   --lambda_decorr 0.5

python run.py --dataset YelpChi --lr 1e-3 --weight_decay 1e-3 --alpha 1.0 --beta 0.1 --lambda_aff 2.0 --lambda_dev 0.5 --lambda_dgi 0.1 --lambda_rec 0.5 --lambda_cross 0.1 --lambda_score 0.5 --lambda_agree 0.2 --w_dgi 0.1 --w_rec 0.1 --wavelet_shape 3.0 --num_filters 4  --runs 10 --device cuda --gpu 2  --lambda_decorr 0.5

python run.py --dataset Amazon  --lr 1e-3  --weight_decay 1e-5  --alpha 0.1 --beta 0.5 --lambda_aff 0.7 --lambda_dev 0.1 --lambda_dgi 1.0 --lambda_rec 0.1 --lambda_cross 0.1 --lambda_score 0.1 --lambda_agree 0.1 --w_dgi 1.0 --w_rec 0.1   --wavelet_shape 0.9  --num_filters 6    --runs 10 --device cuda --gpu 2  --lambda_decorr 0.5

python run.py --dataset disney  --lr 1e-3  --weight_decay 1e-5  --alpha 1.0 --beta 0.1 --lambda_aff 1.0 --lambda_dev 0.1 --lambda_dgi 0.5 --lambda_rec 0.1 --lambda_cross 0.5 --lambda_score 0.1 --lambda_agree 0.1 --w_dgi 1.0 --w_rec 0.1    --wavelet_shape 0.2  --num_filters 3   --runs 10 --device cuda --gpu 2  --lambda_decorr 0.5

python run.py --dataset enron  --lr 6e-5  --weight_decay 1e-5  --alpha 0.1 --beta 1 --lambda_aff 0.1 --lambda_dev 1.0 --lambda_dgi 2.0 --lambda_rec 0.5 --lambda_cross 0.1 --lambda_score 0.8 --lambda_agree 1.0 --w_dgi 2.0 --w_rec 0.1    --wavelet_shape 0.1  --num_filters 4   --runs 10 --device cuda --gpu 2  --lambda_decorr 0.5

python run.py --dataset books  --lr 2e-4  --weight_decay 1e-3  --alpha 0.1 --beta 0.5 --lambda_aff 0.7 --lambda_dev 0.1 --lambda_dgi 0.1 --lambda_rec 0.5 --lambda_cross 0.1 --lambda_score 0.1 --lambda_agree 0.2 --w_dgi 1.0 --w_rec 0.1   --wavelet_shape 0.1  --num_filters 3   --runs 10 --device cuda --gpu 2  --lambda_decorr 0.5