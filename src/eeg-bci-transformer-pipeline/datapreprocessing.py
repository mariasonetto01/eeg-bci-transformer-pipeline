import mne 
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGpreprocessing:
    """
    EEG preprocessing pipeline 
    """
    
    def __init__(self,raw):
        
        self.raw = raw
        self.raw_filtered = None
        self.epochs = None
        self.X_normalized = None 
 
    
    def bandpassfilter(self,l_freq,h_freq,target_sfreq=160.0):
        """
        Ricampiona a 160 Hz Applying a band-pass filter to EEG signals

        Parameters
        ----------
        l_freq : float
            Lower band-pass frequency 
        h_freq : float
            Upper band-pass frequency
        target_sfreq : float
            resamples signal to 160 Hz to ensure uniform length  across all files 
    
        """

        raw_filtered = self.raw.copy()
        
        if raw_filtered.info['sfreq'] != target_sfreq:
            raw_filtered.resample(sfreq=target_sfreq, verbose=False)

        raw_filtered.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design='firwin',
            skip_by_annotation='edge',
            verbose=False
        )

        self.raw_filtered=raw_filtered

        return raw_filtered

    def epoching(self,tmin,tmax): #T1 e T2 non sono sempre left e right, dipende dal numero del run, per la classificazione L/R che prenso di fare io ok
        """
        Extracts target epochs from filtered EEG signals based on task annotations.

        tmin : float
            Start time before cue onset (in seconds).
        tmax : float
            End time after cue onset (in seconds).
    
        """

        events, event_id = mne.events_from_annotations(self.raw_filtered)

        event_dict = {}
        if "T1" in event_id:
            event_dict["Left"] = event_id["T1"]
        if "T2" in event_id:
            event_dict["Right"] = event_id["T2"]
        

        epochs = mne.Epochs(
            self.raw_filtered,          
            events,                
            event_id=event_dict,   # Left and Right only

            tmin=tmin,              
            tmax=tmax,              

            baseline=None,         # No baseline correction
            preload=True
        )

        self.epochs = epochs

        return epochs
    '''
    def normalize_conrumore(self, alpha=0.01, eps=0): #1e-8
        """
        Applica la normalizzazione Z-score globale e aggiunge rumore casuale 
        secondo la formula: X* = (X - mu) / delta + alpha * N.

        Parameters
        ----------
        alpha : float
            Fattore di controllo per la percentuale di rumore casuale (default 0.01)[cite: 2].
        eps : float
            Piccola costante per evitare la divisione per zero.
        """
        # Estrae i dati dalle epoche
        X = self.epochs.get_data(copy=True)

        # Calcola media e deviazione standard globali dei dati
        mean = np.mean(X)
        std = np.std(X)

        # Normalizzazione Z-score
        normalized_data = (X - mean) / (std + eps)

        # Genera rumore casuale N con distribuzione normale standard
        noise = alpha * np.random.randn(*normalized_data.shape)

        # Aggiunge il rumore
        X_norm = normalized_data + noise

        self.X_normalized = X_norm
        
        return X_norm

    def normalize(self,axis=-1,eps=1e-8):
        """
        Applies Z-Score normalization (mean=0, std=1) to the extracted epochs data.

        Parameters
         ----------
        axis : int or tuple
            Axis or axes along which the mean and std are computed. Default is -1 (time axis).
        eps : float
            Small constant to prevent division by zero. Default is 1e-8.
        """
        X = self.epochs.get_data(copy=True)

        mean = np.mean(X, axis=axis, keepdims=True)
        std = np.std(X, axis=axis, keepdims=True)

        X_norm = (X - mean) / (std + eps)

        self.X_normalized = X_norm
        
        return X_norm

'''
class EEGDataset(Dataset): #posso definirlo in maniera generica così lo posso riutilizzare per trainingset,testset e validationset 

    def __init__(self, data, labels):
        """Convert to PyTorch tensors"""
        
        self.x = torch.from_numpy(data).float()
        self.y = torch.from_numpy(labels).long()


    def __getitem__(self, idx):
        '''  # Estrai la finestra temporale e l'etichetta corrispondente'''

        sample_x = self.x[idx]
        sample_y = self.y[idx]

        return sample_x, sample_y

    def __len__(self):
        len_x=len(self.x)

        return len_x
    


