from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer, seed_everything
import sys
import importlib

# sys.path.append('/app/face_lib')
# sys.path.append('/home/i.kolesnikov/face_ue/face_lib')
# code based on original repo: https://github.com/MathsShen/SCF
# main config for training: configs/hydra/train_sphere_face.yaml
# simple demo classes for your convenience
# from face_lib.models.scf import SphereConfidenceFace
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="/app/configs/train/train_hydra/", config_name="train_scf_base.yaml")
def train_model(cfg):
    #cli = LightningCLI(SphereConfidenceFace, parser_kwargs={"parser_mode": "omegaconf"})
    #print(**cfg.model.init_args, type(**cfg.model.init_args))
    seed_everything(cfg.seed_everything, workers=True)

    # backbone = getattr(
    #     importlib.import_module(cfg.model.backbone.class_path),
    #     cfg.model.backbone.backbone_name
    # )(
    #     **cfg.model.backbone.init_args
    # )

    # head = getattr(
    #     importlib.import_module(cfg.model.head.class_path),
    #     cfg.model.head.head_name
    # )(
    #     **cfg.model.head.init_args
    # )

    # scf_loss = getattr(
    #     importlib.import_module(cfg.model.scf_loss.class_path),
    #     cfg.model.scf_loss.loss_name
    # )(
    #     **cfg.model.scf_loss.init_args
    # )

    # softmax_weights = getattr(
    #     importlib.import_module(cfg.model.softmax_weights.class_path),
    #     cfg.model.softmax_weights.sw_name
    # )(
    #     **cfg.model.softmax_weights.init_args
    # )

    # validation_dataset = getattr(
    #     importlib.import_module(cfg.model.validation_dataset.class_path),
    #     cfg.model.validation_dataset.dataset_name
    # )(
    #     **cfg.model.validation_dataset.init_args
    # )

    # template_pooling_strategy = getattr(
    #     importlib.import_module(cfg.model.template_pooling_strategy.class_path),
    #     cfg.model.template_pooling_strategy.strategy_name
    # )()

    # dist_function = getattr(
    #     importlib.import_module(cfg.model.recognition_method.init_args.distance_function.class_path),
    #     cfg.model.recognition_method.init_args.distance_function.dist_name
    # )()
    # recognition_method = getattr(
    #     importlib.import_module(cfg.model.recognition_method.class_path),
    #     cfg.model.recognition_method.method_name
    # )(
    #     dist_function
    # )

    # model = getattr(
    #     importlib.import_module(cfg.model.class_path),
    #     cfg.model.model_name
    # )(backbone, head, scf_loss, OmegaConf.to_container(cfg.model.optimizer_params), OmegaConf.to_container(cfg.model.scheduler_params), cfg.model.init_args.permute_batch, softmax_weights, validation_dataset, template_pooling_strategy, recognition_method, OmegaConf.to_container(cfg.model.init_args.verification_metrics))
    
    # logger = getattr(
    #     importlib.import_module(cfg.trainer.logger.class_path),
    #     cfg.trainer.logger.logger_name
    # )(
    #     **cfg.trainer.logger.init_args
    # )

    # callbacks = []
    # model_checkpoint = getattr(
    #     importlib.import_module(cfg.trainer.callbacks[0].class_path),
    #     cfg.trainer.callbacks[0].callback_name
    # )(
    #     **cfg.trainer.callbacks[0].init_args
    # )
    # learning_rate_monitor = getattr(
    #     importlib.import_module(cfg.trainer.callbacks[1].class_path),
    #     cfg.trainer.callbacks[1].callback_name
    # )(
    #     **cfg.trainer.callbacks[1].init_args
    # )
    # prediction_writer = getattr(
    #     importlib.import_module(cfg.trainer.callbacks[2].class_path),
    #     cfg.trainer.callbacks[2].callback_name
    # )(
    #     **cfg.trainer.callbacks[2].init_args
    # )
    # callbacks.append(model_checkpoint)
    # callbacks.append(learning_rate_monitor)
    # callbacks.append(prediction_writer)

    # inp_dict = OmegaConf.to_container(cfg.trainer.init_args)
    # inp_dict["callbacks"] = callbacks
    # inp_dict["logger"] = logger
    # trainer = Trainer(**inp_dict)

    # train_dataset = getattr(
    #     importlib.import_module(cfg.data.init_args.train_dataset.class_path),
    #     cfg.data.init_args.train_dataset.dataset_name
    # )(
    #     **cfg.data.init_args.train_dataset.init_args
    # )
    # validation_dataset = getattr(
    #     importlib.import_module(cfg.data.init_args.validation_dataset.class_path),
    #     cfg.data.init_args.validation_dataset.dataset_name
    # )(
    #     **cfg.data.init_args.validation_dataset.init_args
    # )
    # predict_dataset = getattr(
    #     importlib.import_module(cfg.data.init_args.predict_dataset.class_path),
    #     cfg.data.init_args.predict_dataset.dataset_name
    # )(
    #     **cfg.data.init_args.predict_dataset.init_args
    # )
    # dataclass = predict_dataset = getattr(
    #     importlib.import_module(cfg.data.class_path),
    #     cfg.data.data_name
    # )(
    #     train_dataset = train_dataset,
    #     validation_dataset = validation_dataset,
    #     predict_dataset = predict_dataset,
    #     batch_size = cfg.data.init_args.batch_size,
    #     num_workers = cfg.data.init_args.num_workers
    # )

    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    dataclass = instantiate(cfg.data)

    trainer.fit(model = model, datamodule=dataclass)

if __name__ == "__main__":
    train_model()