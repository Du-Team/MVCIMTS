def get_default_config(data_name):
    if data_name in ['BDGP']:
        return dict(
            Prediction=dict(
               view0=[1750, 79],
               view1=[1750, 79],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['gait']:
        return dict(
            Prediction=dict(
               view0=[128, 256, 128],
               view1=[128, 256, 128],
               view2=[128, 256, 128],
               view3=[128, 256, 128],
               view4=[128, 256, 128],
               view5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['ArticularyWordRecognition']:
        return dict(
            Prediction=dict(
               view0=[128, 256, 128],
               view1=[128, 256, 128],
               view2=[128, 256, 128],
               view3=[128, 256, 128],
               view4=[128, 256, 128],
               view5=[128, 256, 128],
               view6=[128, 256, 128],
               view7=[128, 256, 128],
               view8=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.7,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['BasicMotions']:
        return dict(
            Prediction=dict(
               view0=[128, 256, 128],
               view1=[128, 256, 128],
               view2=[128, 256, 128],
               view3=[128, 256, 128],
               view4=[128, 256, 128],
               view5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['Epilepsy']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['EthanolConcentration']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['HandMovementDirection']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
                view6=[128, 256, 128],
                view7=[128, 256, 128],
                view8=[128, 256, 128],
                view9=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['PenDigits']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['Handwriting']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['RacketSports']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['Libras']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['UWaveGestureLibrary']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.7,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['LSST']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['StandWalkJump']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['SelfRegulationSCP1']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['SelfRegulationSCP2']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
                view6=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['AtrialFibrillation']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['FingerMovements']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
                view6=[128, 256, 128],
                view7=[128, 256, 128],
                view8=[128, 256, 128],
                view9=[128, 256, 128],
                view10=[128, 256, 128],
                view11=[128, 256, 128],
                view12=[128, 256, 128],
                view13=[128, 256, 128],
                view14=[128, 256, 128],
                view15=[128, 256, 128],
                view16=[128, 256, 128],
                view17=[128, 256, 128],
                view18=[128, 256, 128],
                view19=[128, 256, 128],
                view20=[128, 256, 128],
                view21=[128, 256, 128],
                view22=[128, 256, 128],
                view23=[128, 256, 128],
                view24=[128, 256, 128],
                view25=[128, 256, 128],
                view26=[128, 256, 128],
                view27=[128, 256, 128],

            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['LP1']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],

            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.7,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['LP4']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],

            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['LP5']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],

            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['II_Ia_data']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],

            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['II_Ib_data']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
                view6=[128, 256, 128],

            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['III_V_s2_data']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
                view3=[128, 256, 128],
                view4=[128, 256, 128],
                view5=[128, 256, 128],
                view6=[128, 256, 128],
                view7=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    elif data_name in ['IV_2b_s2_data']:
        return dict(
            Prediction=dict(
                view0=[128, 256, 128],
                view1=[128, 256, 128],
                view2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=0.5,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )
    else:
        raise Exception('Undefined data_name')
