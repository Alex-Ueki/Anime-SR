"""
    DEPRECIATED
"""
import Modules.srcnn as srcnn
import Modules.setup as setup

if __name__ == "__main__":

    """
    Train Super Resolution
    """

    sr = srcnn.BasicSR()
    sr.create_model()
    sr.fit(nb_epochs=250)

    """
    Evaluate BasicSR on eval_images
    """

    # sr = srcnn.BasicSR()
    # sr.evaluate()

    """
    Predict HD images in predict_images using BasicSR
    """

    sr = srcnn.BasicSR()
    sr.predict()

    """
    Train ExpansionSuperResolution
    """

    # esr = srcnn.ExpansionSR()
    # esr.create_model()
    # esr.fit(nb_epochs=250)

    """
    Evaluate ESRCNN on eval_images
    """

    # esr = srcnn.ExpansionSR()
    # esr.evaluate()

    """
    Predict HD images in predict_images using ExpansionSR
    """

    # esr = srcnn.ExpansionSR()
    # esr.predict()

    """
    Train DeepDenoiseSR
    """

    # ddsr = srcnn.DeepDenoiseSR()
    # ddsr.create_model()
    # ddsr.fit(nb_epochs=80)

    """
    Evaluate DDSRCNN on eval_images
    """

    ddsr = srcnn.DeepDenoiseSR()
    ddsr.evaluate()

    """
    Predict HD images in predict_images using DeepDenoiseSR
    """

    #ddsr = srcnn.DeepDenoiseSR()
    #ddsr.predict()

    """
    Train VDSR (Very Deep Super Resolution)
    """

    # vdsr = srcnn.VDSR()
    # vdsr.create_model()
    # vdsr.fit(nb_epochs=80)

    """
    Evaluate VDSR on eval_images
    """

    # vdsr = srcnn.VDSR()
    # vdsr.evaluate()

    """
    Predict HD images in predict_images using VDSR
    """

    #ddsr = srcnn.VDSR()
    #ddsr.predict()
