# cardiac_segmentation.py
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.models import load_model
import cv2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import Counter
import joblib
import pandas as pd

sys.path.append('C:\\Alex The Great\\Project\\medai-env\\Scikit-learn\\session10')

class CardiacDataPipeline:
    def __init__(self, data_loader, target_size=(192, 192)):
        self.data_loader = data_loader
        self.target_size = target_size
    
    def prepare_data(self, samples_indices, view='2ch', phase='ED', batch_size=32, augment=True):
        frames = []
        masks = []
        
        print(f"Preparing data for {len(samples_indices)} samples ({view}, {phase})...")
        
        for idx in samples_indices:
            frame_key = f"{view}_{phase}_frame"
            mask_key = f"{view}_{phase}_mask"
            
            sample_data = self.data_loader.load_patient_data(idx)
            frame = sample_data.get(frame_key)
            mask = sample_data.get(mask_key)
            
            if frame is not None and mask is not None:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                
                frames.append(frame)
                masks.append(mask)
        
        if len(frames) == 0:
            raise ValueError("No valid samples found! Check data loader.")
        
        frames = np.array(frames)[..., np.newaxis]
        masks = np.array(masks)
        
        all_mask_values = masks.flatten()
        class_counts = Counter(all_mask_values)
        print(f"Class distribution: {dict(class_counts)}")
        
        print(f"Data prepared: {frames.shape} frames, {masks.shape} masks")
        
        dataset = tf.data.Dataset.from_tensor_slices((frames, masks))
        dataset = dataset.shuffle(1000)
        
        if augment:
            dataset = dataset.map(self._augment_data_fixed, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_data_fixed(self, frame, mask):
        mask_expanded = tf.expand_dims(mask, axis=-1)
        
        if tf.random.uniform(()) > 0.5:
            frame = tf.image.flip_left_right(frame)
            mask_expanded = tf.image.flip_left_right(mask_expanded)
        
        if tf.random.uniform(()) > 0.5:
            frame = tf.image.flip_up_down(frame)
            mask_expanded = tf.image.flip_up_down(mask_expanded)
        
        angle = tf.random.uniform([], -0.1, 0.1)
        
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        frame = tf.image.rot90(frame, k)
        mask_expanded = tf.image.rot90(mask_expanded, k)
        
        frame = tf.image.random_brightness(frame, max_delta=0.1)
        frame = tf.clip_by_value(frame, 0.0, 1.0)
        
        mask_aug = tf.squeeze(mask_expanded, axis=-1)
        
        return frame, mask_aug

class DeepUNetModel:
    def __init__(self, input_shape=(192, 192, 1), num_classes=4, name="deep_unet"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = self.build_deep_unet()
    
    def build_deep_unet(self):
        inputs = layers.Input(shape=self.input_shape)
        
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        p1 = layers.Dropout(0.1)(p1)
        
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        p2 = layers.Dropout(0.1)(p2)
        
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        p3 = layers.Dropout(0.2)(p3)
        
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
        c4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        p4 = layers.Dropout(0.2)(p4)
        
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
        c5 = layers.BatchNormalization()(c5)
        p5 = layers.MaxPooling2D((2, 2))(c5)
        p5 = layers.Dropout(0.3)(p5)
        
        c6 = layers.Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p5)
        c6 = layers.Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Dropout(0.3)(c6)
        
        u6 = layers.Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(c6)
        u6 = layers.concatenate([u6, c5])
        u6 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u6)
        u6 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u6)
        u6 = layers.BatchNormalization()(u6)
        u6 = layers.Dropout(0.3)(u6)
        
        u5 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(u6)
        u5 = layers.concatenate([u5, c4])
        u5 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u5)
        u5 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u5)
        u5 = layers.BatchNormalization()(u5)
        u5 = layers.Dropout(0.2)(u5)
        
        u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u5)
        u4 = layers.concatenate([u4, c3])
        u4 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u4)
        u4 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u4)
        u4 = layers.BatchNormalization()(u4)
        u4 = layers.Dropout(0.2)(u4)
        
        u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u4)
        u3 = layers.concatenate([u3, c2])
        u3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        u3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        u3 = layers.BatchNormalization()(u3)
        u3 = layers.Dropout(0.1)(u3)
        
        u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u3)
        u2 = layers.concatenate([u2, c1])
        u2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u2 = layers.BatchNormalization()(u2)
        u2 = layers.Dropout(0.1)(u2)
        
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(u2)
        
        model = Model(inputs, outputs, name=self.name)
        return model
    
    def compile(self, learning_rate=1e-3):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Compiled {self.name}: {self.model.count_params():,} parameters")
    
    def train(self, train_dataset, val_dataset, epochs=100, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        model_save_path = os.path.join(model_dir, f'best_{self.name}.h5')
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{self.name}.h5')
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
                save_freq='epoch'
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, f'logs_{self.name}'),
                histogram_freq=1
            )
        ]
        
        print(f"Training {self.name} for {epochs} epochs...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.model.save(model_save_path)
        print(f"FINISHED training {self.name}")
        
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"Best validation accuracy for {self.name}: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return history

class DeepCardiacEnsembleMeta:
    def __init__(self, input_shape=(192, 192, 1), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        self.meta_learner = None
        self.model_dir = f"deep_cardiac_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Model directory: {self.model_dir}")
    
    def build_models(self):
        print("Building Deep U-Net models...")
        
        self.models['deep_unet_v1'] = DeepUNetModel(self.input_shape, self.num_classes, "deep_unet_v1")
        self.models['deep_unet_v2'] = DeepUNetModel(self.input_shape, self.num_classes, "deep_unet_v2")
        
        for name, model in self.models.items():
            model.compile(learning_rate=1e-3)
    
    def train_models(self, train_dataset, val_dataset, epochs=150):
        histories = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model_history = model.train(train_dataset, val_dataset, epochs, self.model_dir)
            histories[name] = model_history
        
        return histories
    
    def extract_features(self, dataset):
        print("Extracting features for meta-learning...")
        
        all_features = []
        all_true_labels = []
        
        frames_list = []
        masks_list = []
        for batch in dataset:
            f, m = batch
            frames_list.append(f.numpy())
            masks_list.append(m.numpy())
        
        if not frames_list:
            print("No data in dataset!")
            return np.array([]), np.array([])
            
        frames = np.concatenate(frames_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)
        
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            all_true_labels.append(dominant_class)
        
        all_true_labels = np.array(all_true_labels)
        
        for name, model_obj in self.models.items():
            print(f"   Processing {name}...")
            model = model_obj.model
            predictions = model.predict(frames, verbose=0, batch_size=16)
            
            features = []
            for pred in predictions:
                class_probs = np.mean(pred, axis=(0, 1))
                max_probs = np.max(pred, axis=(0, 1))
                entropy = -np.sum(pred * np.log(pred + 1e-8))
                confidence = np.mean(max_probs)
                
                std_probs = np.std(pred, axis=(0, 1))
                mean_max_prob = np.mean(np.max(pred, axis=-1))
                
                feature_vector = np.concatenate([
                    class_probs,
                    max_probs,
                    std_probs,
                    [entropy],
                    [confidence],
                    [mean_max_prob]
                ])
                features.append(feature_vector)
            
            all_features.append(np.array(features))
        
        if not all_features:
            print("No features extracted!")
            return np.array([]), np.array([])
            
        combined_features = np.concatenate(all_features, axis=1)
        
        print(f"Features shape: {combined_features.shape}")
        print(f"Labels distribution: {Counter(all_true_labels)}")
        
        return combined_features, all_true_labels
    
    def train_meta_learner(self, train_dataset, val_dataset):
        print("Training Meta-Learner...")
        
        X_train, y_train = self.extract_features(train_dataset)
        X_val, y_val = self.extract_features(val_dataset)
        
        if X_train.size == 0 or X_val.size == 0:
            print("No features for meta-learning!")
            self.meta_learner = 'averaging'
            self.meta_learner_name = 'simple_averaging'
            return 0.0
        
        meta_learners = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        best_score = 0
        best_learner = None
        
        for name, learner in meta_learners.items():
            print(f"Training {name}...")
            try:
                learner.fit(X_train, y_train)
                y_pred = learner.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                print(f"   {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_learner = learner
                    self.meta_learner = learner
                    self.meta_learner_name = name
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        if best_learner is not None:
            meta_path = os.path.join(self.model_dir, 'meta_learner.pkl')
            joblib.dump(best_learner, meta_path)
            print(f"Best meta-learner: {self.meta_learner_name} (acc: {best_score:.4f})")
        else:
            print("No meta-learner trained - using averaging")
            self.meta_learner = 'averaging'
            self.meta_learner_name = 'simple_averaging'
            best_score = 0.0
        
        return best_score
    
    def calculate_dice_scores(self, y_true, y_pred):
        dice_scores = {}
        for class_id in range(self.num_classes):
            true_binary = (y_true == class_id).astype(np.float32)
            pred_binary = (y_pred == class_id).astype(np.float32)
            
            intersection = np.sum(true_binary * pred_binary)
            union = np.sum(true_binary) + np.sum(pred_binary)
            
            if union == 0:
                dice = 1.0
            else:
                dice = (2. * intersection) / (union + 1e-8)
            
            dice_scores[class_id] = dice
        
        return dice_scores
    
    def evaluate_ensemble(self, test_dataset):
        print("Evaluating Ensemble with Dice Scores...")
        
        frames_list = []
        masks_list = []
        for batch in test_dataset:
            f, m = batch
            frames_list.append(f.numpy())
            masks_list.append(m.numpy())
        
        frames = np.concatenate(frames_list, axis=0)
        true_masks = np.concatenate(masks_list, axis=0)
        
        all_predictions = []
        for name, model_obj in self.models.items():
            model = model_obj.model
            pred = model.predict(frames, verbose=0, batch_size=16)
            all_predictions.append(pred)
        
        if self.meta_learner == 'averaging':
            avg_predictions = np.mean(all_predictions, axis=0)
            ensemble_pred = np.argmax(avg_predictions, axis=-1)
        else:
            features, true_labels = self.extract_features(test_dataset)
            if features.size > 0:
                ensemble_pred_labels = self.meta_learner.predict(features)
                ensemble_pred = np.zeros_like(true_masks)
                for i, label in enumerate(ensemble_pred_labels):
                    ensemble_pred[i] = label
            else:
                avg_predictions = np.mean(all_predictions, axis=0)
                ensemble_pred = np.argmax(avg_predictions, axis=-1)
        
        accuracy = accuracy_score(true_masks.flatten(), ensemble_pred.flatten())
        
        dice_scores = self.calculate_dice_scores(true_masks.flatten(), ensemble_pred.flatten())
        mean_dice = np.mean(list(dice_scores.values()))
        
        per_image_dice = []
        for i in range(len(true_masks)):
            img_dice = self.calculate_dice_scores(true_masks[i].flatten(), ensemble_pred[i].flatten())
            per_image_dice.append(img_dice)
        
        mean_per_image_dice = np.mean([np.mean(list(dice.values())) for dice in per_image_dice])
        
        results = {
            'accuracy': accuracy,
            'mean_dice': mean_dice,
            'mean_per_image_dice': mean_per_image_dice,
            'dice_scores': dice_scores,
            'per_image_dice': per_image_dice,
            'meta_learner': self.meta_learner_name
        }
        
        print(f"ENSEMBLE RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Mean Dice (global): {mean_dice:.4f}")
        print(f"Mean Dice (per image): {mean_per_image_dice:.4f}")
        print(f"Meta-Learner: {self.meta_learner_name}")
        
        print(f"Dice Scores per Class:")
        class_names = ['Background', 'LV', 'Myocardium', 'RV']
        for i, class_name in enumerate(class_names):
            dice_score = dice_scores[i]
            print(f"   {class_name}: {dice_score:.4f}")
        
        self._plot_dice_scores(dice_scores, class_names)
        self._save_results(results)
        
        return results
    
    def _plot_dice_scores(self, dice_scores, class_names):
        plt.figure(figsize=(10, 6))
        dice_values = [dice_scores[i] for i in range(len(class_names))]
        
        colors = ['gray', 'red', 'green', 'blue']
        bars = plt.bar(class_names, dice_values, color=colors, alpha=0.8)
        
        plt.ylabel('Dice Coefficient', fontsize=12)
        plt.title('Dice Scores per Class - Deep U-Net Ensemble', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, dice in zip(bars, dice_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{dice:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'dice_scores_per_class.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_results(self, results):
        results_data = {
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'model_dir': self.model_dir,
            'models': list(self.models.keys()),
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        with open(os.path.join(self.model_dir, 'final_results.json'), 'w') as f:
            json.dump(results_data, f, indent=4)
        
        print(f"Results saved to {self.model_dir}")

class CardiacModelEvaluator:
    def __init__(self, model_dir, data_loader):
        self.model_dir = model_dir
        self.data_loader = data_loader
        self.models = {}
        self.meta_learner = None
        self.load_models()
    
    def load_models(self):
        print("Loading trained models...")
        
        model_files = {}
        for file in os.listdir(self.model_dir):
            if file.endswith('.h5') and 'best' in file:
                model_name = file.replace('best_', '').replace('.h5', '')
                model_files[model_name] = os.path.join(self.model_dir, file)
        
        for name, path in model_files.items():
            try:
                self.models[name] = load_model(path)
                print(f"Loaded {name}: {path}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        meta_path = os.path.join(self.model_dir, 'meta_learner.pkl')
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            print(f"Loaded meta-learner: {meta_path}")
        else:
            print("No meta-learner found, using averaging")
    
    def prepare_single_sample(self, patient_idx, view='2ch', phase='ED', target_size=(192, 192)):
        sample_data = self.data_loader.load_patient_data(patient_idx)
        
        frame_key = f"{view}_{phase}_frame"
        mask_key = f"{view}_{phase}_mask"
        
        frame = sample_data.get(frame_key)
        mask = sample_data.get(mask_key)
        
        if frame is None or mask is None:
            print(f"No data for patient {patient_idx}, {view}, {phase}")
            return None, None
        
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        frame_resized = (frame_resized - frame_resized.min()) / (frame_resized.max() - frame_resized.min() + 1e-8)
        
        frame_final = np.expand_dims(np.expand_dims(frame_resized, axis=-1), axis=0)
        mask_final = np.expand_dims(mask_resized, axis=0)
        
        return frame_final, mask_final
    
    def analyze_data_distribution(self, indices, view='2ch', phase='ED'):
        print("Analyzing real data distribution...")
        
        all_classes = []
        class_distributions = []
        
        for idx in indices:
            sample_data = self.data_loader.load_patient_data(idx)
            mask_key = f"{view}_{phase}_mask"
            mask = sample_data.get(mask_key)
            
            if mask is not None:
                unique_vals, counts = np.unique(mask, return_counts=True)
                all_classes.extend(unique_vals)
                
                dist = dict(zip(unique_vals, counts))
                class_distributions.append(dist)
                
                if len(unique_vals) > 1:
                    print(f"Patient {idx}: classes {list(unique_vals)} - distribution {dist}")
        
        class_counter = Counter(all_classes)
        print(f"OVERALL CLASS DISTRIBUTION:")
        print("=" * 50)
        class_names = ['Background', 'LV', 'Myocardium', 'RV']
        for class_id in range(4):
            count = class_counter.get(class_id, 0)
            percentage = (count / len(all_classes)) * 100 if all_classes else 0
            print(f"Class {class_id} ({class_names[class_id]}): {count} occurrences ({percentage:.2f}%)")
        
        return class_counter, class_distributions
    
    def calculate_dice_scores_per_class(self, y_true, y_pred):
        dice_scores = {}
        
        for class_id in range(4):
            true_binary = (y_true == class_id).astype(np.float32)
            pred_binary = (y_pred == class_id).astype(np.float32)
            
            intersection = np.sum(true_binary * pred_binary)
            union = np.sum(true_binary) + np.sum(pred_binary)
            
            if union == 0:
                dice = 1.0 if np.sum(true_binary) == 0 else 0.0
            else:
                dice = (2. * intersection) / (union + 1e-8)
            
            dice_scores[class_id] = dice
        
        return dice_scores
    
    def evaluate_on_test_set(self, test_indices, view='2ch', phase='ED'):
        print(f"Evaluating on {len(test_indices)} test samples...")
        
        all_true_masks = []
        all_pred_masks = []
        all_pred_probs = []
        
        for idx in test_indices:
            frame, true_mask = self.prepare_single_sample(idx, view, phase)
            
            if frame is None:
                continue
            
            model_predictions = []
            for name, model in self.models.items():
                pred = model.predict(frame, verbose=0)
                model_predictions.append(pred)
            
            if len(model_predictions) > 0:
                avg_pred = np.mean(model_predictions, axis=0)
                pred_mask = np.argmax(avg_pred, axis=-1)[0]
                pred_prob = avg_pred[0]
                
                all_true_masks.append(true_mask[0])
                all_pred_masks.append(pred_mask)
                all_pred_probs.append(pred_prob)
        
        if not all_true_masks:
            print("No valid predictions!")
            return {}
        
        true_masks = np.array(all_true_masks)
        pred_masks = np.array(all_pred_masks)
        pred_probs = np.array(all_pred_probs)
        
        print(f"True masks shape: {true_masks.shape}")
        print(f"Pred masks shape: {pred_masks.shape}")
        print(f"Pred probs shape: {pred_probs.shape}")
        
        results = self.calculate_comprehensive_metrics(true_masks, pred_masks, pred_probs)
        return results
    
    def calculate_comprehensive_metrics(self, true_masks, pred_masks, pred_probs):
        print("Calculating comprehensive metrics...")
        
        dice_scores_per_class = {}
        mean_dice_per_image = []
        
        for i in range(len(true_masks)):
            dice_scores = self.calculate_dice_scores_per_class(true_masks[i].flatten(), pred_masks[i].flatten())
            mean_dice_per_image.append(np.mean(list(dice_scores.values())))
            
            for class_id, dice in dice_scores.items():
                if class_id not in dice_scores_per_class:
                    dice_scores_per_class[class_id] = []
                dice_scores_per_class[class_id].append(dice)
        
        mean_dice_global = np.mean([np.mean(scores) for scores in dice_scores_per_class.values()])
        mean_dice_per_class = {class_id: np.mean(scores) for class_id, scores in dice_scores_per_class.items()}
        
        accuracy = np.mean(true_masks.flatten() == pred_masks.flatten())
        
        cm = confusion_matrix(true_masks.flatten(), pred_masks.flatten(), labels=[0, 1, 2, 3])
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        true_flat = true_masks.flatten()
        pred_probs_flat = pred_probs.reshape(-1, 4)
        
        for class_id in range(4):
            true_binary = (true_flat == class_id).astype(int)
            pred_binary = pred_probs_flat[:, class_id]
            
            fpr[class_id], tpr[class_id], _ = roc_curve(true_binary, pred_binary)
            roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])
        
        results = {
            'accuracy': accuracy,
            'mean_dice_global': mean_dice_global,
            'mean_dice_per_class': mean_dice_per_class,
            'dice_scores_per_class': dice_scores_per_class,
            'mean_dice_per_image': mean_dice_per_image,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'true_masks': true_masks,
            'pred_masks': pred_masks,
            'pred_probs': pred_probs
        }
        
        return results
    
    def plot_comprehensive_results(self, results, output_dir=None):
        if output_dir is None:
            output_dir = self.model_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        class_names = ['Background', 'LV', 'Myocardium', 'RV']
        colors = ['gray', 'red', 'green', 'blue']
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        dice_values = [results['mean_dice_per_class'].get(i, 0) for i in range(4)]
        bars = plt.bar(class_names, dice_values, color=colors, alpha=0.8)
        plt.ylabel('Dice Coefficient')
        plt.title('Mean Dice Scores per Class', fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, dice in zip(bars, dice_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{dice:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(2, 2, 2)
        for class_id in range(4):
            if class_id in results['roc_auc']:
                plt.plot(results['fpr'][class_id], results['tpr'][class_id],
                        color=colors[class_id], lw=2,
                        label=f'{class_names[class_id]} (AUC = {results["roc_auc"][class_id]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves per Class', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix', fontweight='bold')
        
        plt.subplot(2, 2, 4)
        plt.hist(results['mean_dice_per_image'], bins=20, alpha=0.7, color='purple')
        plt.xlabel('Dice Coefficient')
        plt.ylabel('Frequency')
        plt.title('Distribution of Dice Scores per Image', fontweight='bold')
        plt.grid(alpha=0.3)
        
        mean_dice = np.mean(results['mean_dice_per_image'])
        plt.axvline(mean_dice, color='red', linestyle='--', label=f'Mean: {mean_dice:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Absolute Numbers)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_absolute.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 8))
        for class_id in range(4):
            if class_id in results['roc_auc']:
                plt.plot(results['fpr'][class_id], results['tpr'][class_id],
                        color=colors[class_id], lw=3,
                        label=f'{class_names[class_id]} (AUC = {results["roc_auc"][class_id]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Classes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves_detailed.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_results(self, results):
        class_names = ['Background', 'LV', 'Myocardium', 'RV']
        
        print("="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Global Mean Dice: {results['mean_dice_global']:.4f}")
        print(f"Mean Dice per Image: {np.mean(results['mean_dice_per_image']):.4f}")
        
        print(f"Dice Scores per Class:")
        for class_id in range(4):
            dice = results['mean_dice_per_class'].get(class_id, 0)
            print(f"   {class_names[class_id]}: {dice:.4f}")
        
        print(f"AUC Scores per Class:")
        for class_id in range(4):
            auc_score = results['roc_auc'].get(class_id, 0)
            print(f"   {class_names[class_id]}: {auc_score:.4f}")
        
        print(f"Class Distribution in Predictions:")
        pred_dist = Counter(results['pred_masks'].flatten())
        for class_id in range(4):
            count = pred_dist.get(class_id, 0)
            percentage = (count / len(results['pred_masks'].flatten())) * 100
            print(f"   {class_names[class_id]}: {count} pixels ({percentage:.2f}%)")
        
        print(f"Class Distribution in Ground Truth:")
        true_dist = Counter(results['true_masks'].flatten())
        for class_id in range(4):
            count = true_dist.get(class_id, 0)
            percentage = (count / len(results['true_masks'].flatten())) * 100
            print(f"   {class_names[class_id]}: {count} pixels ({percentage:.2f}%)")

def main_training():
    print("Starting Deep U-Net + Meta-Learner Cardiac Segmentation...")
    
    try:
        from data_loaders.camus_hdf5_loader_fixed import CamusHDF5LoaderFixed
        data_loader = CamusHDF5LoaderFixed()
        print("Data loader loaded successfully")
        
        total_samples = 450
        indices = list(range(total_samples))
        
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        data_pipeline = CardiacDataPipeline(data_loader, target_size=(192, 192))
        
        print("Preparing training data...")
        train_dataset = data_pipeline.prepare_data(train_indices, view='2ch', phase='ED', batch_size=8, augment=True)
        
        print("Preparing validation data...")
        val_dataset = data_pipeline.prepare_data(val_indices, view='2ch', phase='ED', batch_size=8, augment=False)
        
        print("Preparing test data...")
        test_dataset = data_pipeline.prepare_data(test_indices, view='2ch', phase='ED', batch_size=8, augment=False)
        
        ensemble = DeepCardiacEnsembleMeta(input_shape=(192, 192, 1), num_classes=4)
        ensemble.build_models()
        
        print("="*60)
        print("Training Deep U-Net Models (150 Epochs)...")
        print("="*60)
        
        histories = ensemble.train_models(train_dataset, val_dataset, epochs=150)
        print("Deep U-Net models training completed")
        
        print("="*60)
        print("Training Meta-Learner...")
        print("="*60)
        
        meta_accuracy = ensemble.train_meta_learner(train_dataset, val_dataset)
        print(f"Meta-learner training completed (accuracy: {meta_accuracy:.4f})")
        
        print("="*60)
        print("Final Evaluation with Dice Scores...")
        print("="*60)
        
        results = ensemble.evaluate_ensemble(test_dataset)
        
        print("DEEP U-NET + META-LEARNER TRAINING COMPLETED!")
        print("="*60)
        
        mean_dice = results['mean_dice']
        mean_per_image_dice = results['mean_per_image_dice']
        
        print(f"FINAL DICE SCORES:")
        print(f"   Global Mean Dice: {mean_dice:.4f}")
        print(f"   Per-Image Mean Dice: {mean_per_image_dice:.4f}")
        
        if mean_dice > 0.90:
            print("EXCELLENT! Mean Dice > 0.90 - Ready for clinical use!")
        elif mean_dice > 0.85:
            print("VERY GOOD! Mean Dice > 0.85 - Excellent for research")
        elif mean_dice > 0.80:
            print("GOOD! Mean Dice > 0.80 - Suitable for medical applications")
        else:
            print("Needs improvement for medical use")
        
        print(f"All results saved in: {ensemble.model_dir}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

def main_evaluation():
    print("Starting Comprehensive Cardiac Model Evaluation...")
    
    try:
        from data_loaders.camus_hdf5_loader_fixed import CamusHDF5LoaderFixed
        data_loader = CamusHDF5LoaderFixed()
        print("Data loader loaded successfully")
        
        model_dir = "C:\\Alex The Great\\Project\\deep_cardiac_ensemble_20251011_120443"
        
        evaluator = CardiacModelEvaluator(model_dir, data_loader)
        
        test_indices = list(range(50, 100))
        class_dist, _ = evaluator.analyze_data_distribution(test_indices)
        
        results = evaluator.evaluate_on_test_set(test_indices)
        
        if results:
            evaluator.print_detailed_results(results)
            evaluator.plot_comprehensive_results(results)
            
            print(f"Evaluation completed! Results saved in: {model_dir}")
        else:
            print("No results to show!")
            
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_training()