import gradio as gr


def classify_image(inp):
    return inp
    # return model.predict(inp)


image = gr.inputs.Image(label="Input Image", source="webcam")
label = gr.outputs.Image()  # gr.outputs.Label(num_top_classes=k)

gr.Interface(fn=classify_image, live=True, inputs=image, outputs=label).launch()
