
import unittest
import torch
import pandas as pd
import tempfile
import shutil
import os
import shutil
from unittest.mock import MagicMock, patch
import numpy as np
from src.data.dataset import NIHChestXRayDataset
from src.training.trainer import Trainer

class TestRefactor(unittest.TestCase):
    def setUp(self):
        # Temp dir for dummy data
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.test_dir, 'images')
        os.makedirs(self.images_dir)
        
        # Create dummy images
        self.image_names = [f'0000000{i}_000.png' for i in range(1, 21)]
        for name in self.image_names:
            with open(os.path.join(self.images_dir, name), 'wb') as f:
                f.write(os.urandom(100)) # Dummy content
                
        # Create dummy CSV
        # 10 'No Finding', 10 'Infiltration'
        data = {
            'Image Index': self.image_names,
            'Finding Labels': ['No Finding'] * 10 + ['Infiltration'] * 10
        }
        self.csv_path = os.path.join(self.test_dir, 'Data_Entry_2017.csv')
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataset_undersampling(self):
        print("\nTesting Dataset Undersampling...")
        # Load dataset with undersampling 0.5 for No Finding (should keep ~5 of 10)
        # We need to mock Image.open because actual files are garbage
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = MagicMock(convert=lambda x: torch.zeros(3, 224, 224))
            
            ds = NIHChestXRayDataset(
                data_dir=self.test_dir, 
                csv_file='Data_Entry_2017.csv', 
                images_dir='images',
                no_finding_keep_frac=0.5
            )
            
            df = ds.df
            no_finding_count = len(df[df['Finding Labels'] == 'No Finding'])
            finding_count = len(df[df['Finding Labels'] != 'No Finding'])
            
            print(f"Original No Finding: 10. After 0.5 undersample: {no_finding_count}")
            print(f"Original Finding: 10. After keeps: {finding_count}")
            
            self.assertTrue(3 <= no_finding_count <= 7, "Undersampling should keep roughly half")
            self.assertEqual(finding_count, 10, "Should keep all findings")

    def test_weight_calculation(self):
        print("\nTesting Weight Calculation...")
        with patch('PIL.Image.open'):
            ds = NIHChestXRayDataset(
                data_dir=self.test_dir,
                csv_file='Data_Entry_2017.csv',
                images_dir='images', 
                no_finding_keep_frac=1.0
            ) 
            weights = ds.get_pos_weight()
            print(f"Calculated weights shape: {weights.shape}")
            self.assertEqual(len(weights), 14)
            # Infiltration has 10 samples, No Finding has 10 samples.
            # Infiltration is index 3 in dataset.all_labels usually, check logic?
            # actually we can just check there are no NaNs or Infs
            self.assertTrue(torch.all(torch.isfinite(weights)))

    def test_trainer_auc(self):
        print("\nTesting Trainer AUC Calculation...")
        # Mock everything
        model = MagicMock()
        # forward returns (batch, 14)
        model.return_value = torch.randn(4, 14)
        model.parameters.return_value = [torch.randn(1)]
        
        train_loader = MagicMock()
        train_loader.__len__.return_value = 2
        # iterable
        train_loader.__iter__.return_value = iter([
            (torch.randn(4, 3, 224, 224), torch.randn(4, 14)),
            (torch.randn(4, 3, 224, 224), torch.randn(4, 14))
        ])
        train_loader.dataset = [0]*8
        
        val_loader = MagicMock()
        val_loader.__len__.return_value = 2
        # Create fake labels with at least one positive to avoid AUC error if all neg/all pos
        # But roc_auc_score handles full arrays.
        labels_batch = torch.zeros(4, 14)
        labels_batch[0, 0] = 1
        labels_batch[1, 1] = 1
        
        val_loader.__iter__.return_value = iter([
            (torch.randn(4, 3, 224, 224), labels_batch),
            (torch.randn(4, 3, 224, 224), labels_batch)
        ])
        val_loader.dataset = [0]*8
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = MagicMock()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
        
        # Test Validate
        loss, auc = trainer.validate(1)
        print(f"Validation Loss: {loss}, AUC: {auc}")
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(auc, float)
        self.assertTrue(0 <= auc <= 1.0)

if __name__ == '__main__':
    unittest.main()
