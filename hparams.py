

class hparams:

    # option parameters

    # Input type:
    # 1. raw [-1, 1]
    # 2. mixture [-1, 1]
    # 3. bits [0, 512]
    # 4. mulaw[0, mulaw_quantize_channels]
    #
    input_type = 'raw'
    #
    # distribution type, currently supports only 'beta' and 'mixture'
    distribution = 'gaussian' # or "mixture"
    log_scale_min = -32.23619130191664 # = float(np.log(1e-7))
    quantize_channels = 65536 # quantize channel used for compute loss for mixture of logistics
    #
    # for Fatcord's original 9 bit audio, specify the audio bit rate. Note this corresponds to network output
    # of size 2**bits, so 9 bits would be 512 output, etc.
    bits = 10
    # for mu-law
    mulaw_quantize_channels = 512
    # note: DCTTS preprocessing is used instead of Fatcord's original.
    #--------------     
    # audio processing parameters
    sample_rate = 22050
    preemphasis = 0.97
    n_fft = 2048
    frame_shift = 0.011609 # seconds
    frame_length = 0.04643  # seconds
    hop_length = 256 #int(sample_rate * frame_shift)  # samples. =256.
    hop_size = hop_length 
    #win_length  = 1024
    win_length = 1024#int(sample_rate * frame_length)  # samples. =1024.
    n_mels = 80  # Number of Mel banks to generate
    num_mels = n_mels
    max_db = 100
    ref_db = 20
    r=1 # reduction factor 1 for not use
    #----------------
    #
    #----------------
    # model parameters
    rnn_dims = 800
    fc_dims = 512
    pad = 2
    # note upsample factors must multiply out to be equal to hop_size, so adjust
    # if necessary (i.e 4 x 4 x 16 = 256)
    upsample_factors = (4, 4, 16)
    compute_dims = 128
    res_out_dims = 128
    res_blocks = 10
    #----------------
    #
    #----------------
    # training parameters
    batch_size = 128
    nepochs = 5000
    save_every_step = 10000
    evaluate_every_step = 5000
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor = 5
    seq_len = seq_len_factor * hop_size
    grad_norm = 10
    #learning rate parameters
    initial_learning_rate=1e-3
    lr_schedule_type = 'step' # or 'noam'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.5
    lr_step_interval = 15000

    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False
    weight_decay = 0.0
    fix_learning_rate = None # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    #-----------------
    
class hparams_bits:
    # option parameters
    # Input type:
    # 1. raw [-1, 1]
    # 2. mixture [-1, 1]
    # 3. bits [0, 512]
    # 4. mulaw[0, mulaw_quantize_channels]
    #
    input_type = 'bits'
    #
    # distribution type, currently supports only 'beta' and 'mixture'
    distribution = 'gaussian' # or "mixture"
    log_scale_min = -32.23619130191664 # = float(np.log(1e-7))
    quantize_channels = 65536 # quantize channel used for compute loss for mixture of logistics
    #
    # for Fatcord's original 9 bit audio, specify the audio bit rate. Note this corresponds to network output
    # of size 2**bits, so 9 bits would be 512 output, etc.
    bits = 10
    # for mu-law
    mulaw_quantize_channels = 512
    # note: DCTTS preprocessing is used instead of Fatcord's original.
    #--------------     
    # audio processing parameters
    sample_rate = 22050
    preemphasis = 0.97
    n_fft = 2048
    frame_shift = 0.011609 # seconds
    frame_length = 0.04643  # seconds
    hop_length = 256 #int(sample_rate * frame_shift)  # samples. =256.
    hop_size = hop_length 
    #win_length  = 1024
    win_length = 1024#int(sample_rate * frame_length)  # samples. =1024.
    n_mels = 80  # Number of Mel banks to generate
    num_mels = n_mels
    max_db = 100
    ref_db = 20
    r=1 # reduction factor 1 for not use
    #----------------
    #
    #----------------
    # model parameters
    rnn_dims = 800
    fc_dims = 512
    pad = 2
    # note upsample factors must multiply out to be equal to hop_size, so adjust
    # if necessary (i.e 4 x 4 x 16 = 256)
    upsample_factors = (4, 4, 16)
    compute_dims = 128
    res_out_dims = 128
    res_blocks = 10
    #----------------
    #
    #----------------
    # training parameters
    batch_size = 32
    nepochs = 5000
    save_every_step = 10000
    evaluate_every_step = 5000
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor = 5
    seq_len = seq_len_factor * hop_size
    grad_norm = 10
    #learning rate parameters
    initial_learning_rate=1e-4
    lr_schedule_type = 'step' # or 'noam'
    # for noam learning rate schedule
    noam_warm_up_steps = 2000 * (batch_size // 16)
    # for step learning rate schedule
    step_gamma = 0.5
    lr_step_interval = 15000

    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False
    weight_decay = 0.0
    fix_learning_rate = 5e-6 # modify if one wants to use a fixed learning rate, else set to None to use noam learning rate
    #-----------------



