pip install cookiecutter dvc
yes | cookiecutter https://github.com/HelmholtzAI-Consultants-Munich/Quicksetup-ai.git --replay
cd Quicksetup-ai/
pip install -e .
python scripts/train.py trainer.min_epochs=1 trainer.max_epochs=2 log_dir=test_case
python scripts/test.py ckpt_path=logs/checkpoints/last.ckpt
