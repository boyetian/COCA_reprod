# RUN SINGLE TESTS

python ./scripts/run_single_model_accuracy.py --config configs/resnet50_resnet18.yaml
python ./scripts/run_single_model_accuracy.py --config configs/vit_base_mobilvit.yaml

# RUN COCA

python main.py --config configs/resnet50_resnet18.yaml --corruption all
python main.py --config configs/vit_base_mobilvit.yaml --corruption all