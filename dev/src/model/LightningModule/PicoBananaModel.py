# Pico banana model

from src.config.libraries import *

class PicobananaModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        # Atributos generales
        self.learning_rate = learning_rate
        self.model = model

        # Función de pérdida para cada tarea
        self.loss_fn = nn.CrossEntropyLoss()

        # # Atributos para almacenar las métricas calculadas para el accuracy para el conjunto de entrenamiento
        # self.train_model_acc = torchmetrics.Accuracy(task="multiclass", num_classes=N_CAR_MODEL_CLASSES)
        # self.train_color_acc = torchmetrics.Accuracy(task="multiclass", num_classes=N_CAR_COLOR_CLASSES)

        # # Atributos para almacenar las métricas calculadas para el accuracy para el conjunto de validación
        # self.val_model_acc = torchmetrics.Accuracy(task="multiclass", num_classes=N_CAR_MODEL_CLASSES)
        # self.val_color_acc = torchmetrics.Accuracy(task="multiclass", num_classes=N_CAR_COLOR_CLASSES)

    def forward(self, x):
        model_out, color_out = self.model(x)
        return model_out, color_out

    def training_step(self, batch, batch_idx):
        # Obtener batch del dataloader del conjunto de entrenamiento
        features, model_labels, color_labels = batch

        # Obtener una predicción para el batch actual 
        model_out, color_out = self(features)

        # Calcular el error para cada una de las categorías deseadas (modelo y color del auto)
        loss_model = self.loss_fn(model_out, model_labels)
        loss_color = self.loss_fn(color_out, color_labels)

        # El error general del batch se considera como la suma de los dos anteriores
        loss = loss_model + loss_color

        # Loggear pérdidas
        self.log("train_model_loss", loss_model, on_epoch=True, logger=True)
        self.log("train_color_loss", loss_color, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Guardar el calculo del Accuracy del entrenamiento
        model_preds = torch.argmax(model_out, dim=1)
        color_preds = torch.argmax(color_out, dim=1)

        self.train_model_acc(model_preds, model_labels)
        self.train_color_acc(color_preds, color_labels)

        self.log("train_model_acc", self.train_model_acc, on_epoch=True, logger=True)
        self.log("train_color_acc", self.train_color_acc, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Obtener batch del dataloader del conjunto de validación
        features, model_labels, color_labels = batch

        # Obtener una predicción para el batch actual 
        model_out, color_out = self(features)

        # Calcular el error para cada una de las categorías deseadas (modelo y color del auto)
        loss_model = self.loss_fn(model_out, model_labels)
        loss_color = self.loss_fn(color_out, color_labels)

        # El error general del batch se considera como la suma de los dos anteriores
        loss = loss_model + loss_color

        # Loggear pérdidas
        self.log("val_model_loss", loss_model, on_epoch=True, logger=True)
        self.log("val_color_loss", loss_color, on_epoch=True, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Guardar el calculo del Accuracy de validación
        model_preds = torch.argmax(model_out, dim=1)
        color_preds = torch.argmax(color_out, dim=1)

        self.val_model_acc(model_preds, model_labels)
        self.val_color_acc(color_preds, color_labels)

        self.log("val_model_acc", self.val_model_acc, on_epoch=True, logger=True)
        self.log("val_color_acc", self.val_color_acc, on_epoch=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        # Configurar optimizador como 
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        return optimizer