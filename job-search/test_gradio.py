import gradio as gr
def button_clicked():
    return "Button was clicked!"


# This is where the interface is defined
with gr.Blocks() as app:
    button=gr.Button("Click me!")
    output= gr.Textbox()
    output1=gr.Textbox()

    button.click(button_clicked, outputs={output,output1})

app.launch()