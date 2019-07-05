    def show_results(response):
        if response.status==200 or response.status==0:
            document["result"].html = response.text
        else:
            warnings.warn(response.text)

    def get_prediction(ev):
        """Get the predicted probability."""
        req = ajax.ajax()
        req.bind('complete', show_results)
        req.open('POST', '/predict', True)
        req.set_header('content-type','application/json')
        data = json.dumps({'user_input': document['user_input'].value})
        # print(data)
        req.send(data)

    document["predict_button"].bind("click", get_prediction)

    def show_recommendations(response):
        if response.status==200 or response.status==0:
            document["recommendations"].html = response.text
        else:
            warnings.warn(response.text)