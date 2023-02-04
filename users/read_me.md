
## User embedding:

```bash
python train_stylometric_for_users
python generate_stylometrics_for_users
python generate_user_view
python users/user_wgcca.py --input user_embeddings/users_view_vectors.csv --output user_embed
dings/users_gcca_embeddings.npz --k 100 --no_of_views 1
```