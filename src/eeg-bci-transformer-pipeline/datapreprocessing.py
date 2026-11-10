import mne 
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGpreprocessing:
    """
    Pipeline for EEG signal preprocessing, including resampling,band-pass filtering, epoch extraction, and normalization.   
    """
    
    def __init__(self,raw):
        
        self.raw = raw
        self.raw_filtered = None
        self.epochs = None
        self.X_normalized = None 
 
    
    def bandpassfilter(self,l_freq,h_freq,target_sfreq=160.0):
        """ 
        Resamples the raw EEG signal to a uniform sampling frequency and applies a band-pass filter.

        Parameters
        ----------
        l_freq : float
            Lower cut-off frequency of the band-pass filter (in Hz).
        h_freq : float
            Upper cut-off frequency of the band-pass filter (in Hz).
        target_sfreq : float, optional
            Target sampling frequency to ensure uniform length across files (default: 160.0 Hz).
    
        Returns
        -------
        raw_filtered : mne.io.Raw
            The processed MNE Raw object after resampling and filtering.
        
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

    def epoching(self,tmin,tmax): 
        """
        Extracts target epochs from the filtered EEG signals based on task annotations.

        Note: Specifically designed for motor imagery runs (e.g., runs 4, 8, and 12), 
        where the annotations 'T1' and 'T2' correspond to Left and Right hand motor imagery, respectively.

        Parameters
        ----------
        tmin : float
            Start time before cue onset (in seconds).
        tmax : float
            End time after cue onset (in seconds).
    
        Returns
        -------
        epochs : mne.Epochs
            The extracted epochs containing only Left and Right classes.
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
    
   
class EEGDataset(Dataset): 
    """
    Custom PyTorch Dataset for EEG data, designed to be generically reusable across training, validation, and test sets.
    """

    def __init__(self, data, labels):
        """
        Initializes the dataset by converting NumPy arrays into PyTorch tensors.

        Parameters
        ----------
        data : np.ndarray
            The input EEG epochs feature data.
        labels : np.ndarray
            The corresponding class labels.

        """
        
        self.x = torch.from_numpy(data).float()
        self.y = torch.from_numpy(labels).long()


    def __getitem__(self, idx):
        """
        Retrieves the sample and its corresponding label at the specified index.
        """

        sample_x = self.x[idx]
        sample_y = self.y[idx]

        return sample_x, sample_y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        len_x=len(self.x)

        return len_x
    


