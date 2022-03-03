# Dataset setup

## Human3.6M

### Setup from preprocessed dataset
Please download the dataset from [Google Drive](https://drive.google.com/drive/folders/1c7Iz6Tt7qbaw0c1snKgcGOD-JGSzuZ4X?usp=sharing). This is the dataset preprocessed by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline).

Then run the following:

```sh
cd data
python -m prepare_data_h36m --from-archive h36m.zip
python prepare_data_2d_h36m_sh.py -pt h36m.zip
cd ..
```
