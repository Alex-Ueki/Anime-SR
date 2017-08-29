
import srcnn

if __name__ == "__main__":

    """
        Usage Notes: Format_type=1 is .png
        default is format_type = 0, which is .dpx
        dpx is currently untested.

        To test dpx, just remove "format_type=1"
    """
    """
    Train Super Resolution
    """

    sr = srcnn.BasicSR(format_type=1)
    sr.create_model()
    sr.fit(nb_epochs=1)

    """
    Evaluate BasicSR on eval_images
    """

    sr = srcnn.BasicSR(format_type=1)
    sr.evaluate()

    """
    Predict HD images in predict_images using BasicSR
    """

    # sr = srcnn.BasicSR(format_type=1)
    # sr.predict()

    """
    Train ExpansionSuperResolution
    """

    # esr = srcnn.ExpansionSR(format_type=1)
    # esr.create_model()
    # esr.fit(nb_epochs=250)

    """
    Evaluate ESRCNN on eval_images
    """

    # esr = srcnn.ExpansionSR(format_type=1)
    # esr.evaluate()

    """
    Predict HD images in predict_images using ExpansionSR
    """

    # esr = srcnn.ExpansionSR(format_type=1)
    # esr.predict()

    """
    Train DeepDenoiseSR
    """

    # ddsr = srcnn.DeepDenoiseSR(format_type=1)
    # ddsr.create_model()
    # ddsr.fit(nb_epochs=2)

    """
    Evaluate DDSRCNN on eval_images
    """

    # ddsr = srcnn.DeepDenoiseSR(format_type=1)
    # ddsr.evaluate()

    """
    Predict HD images in predict_images using DeepDenoiseSR
    """

    # ddsr = srcnn.DeepDenoiseSR(format_type=1)
    # ddsr.predict()

    """
    Train VDSR (Very Deep Super Resolution)
    """

    # vdsr = srcnn.VDSR(format_type=1)
    # vdsr.create_model()
    # vdsr.fit(nb_epochs=80)

    """
    Evaluate VDSR on eval_images
    """

    # vdsr = srcnn.VDSR(format_type=1)
    # vdsr.evaluate()

    """
    Predict HD images in predict_images using VDSR
    """

    #ddsr = srcnn.VDSR(format_type=1)
    #ddsr.predict()
