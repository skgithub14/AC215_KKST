import { BASE_API_URL } from "./Common";

const axios = require('axios');

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    Predict: async function (formData) {
        return await axios.post(BASE_API_URL + "/predict", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
    Predict_prefix: async function (formData) {
        return await axios.post(BASE_API_URL + "/predict_prefix", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
    Predict_distill: async function (formData) {
        return await axios.post(BASE_API_URL + "/predict_distill", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
    Predict_RNN: async function (formData) {
        return await axios.post(BASE_API_URL + "/predict_rnn", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
}

export default DataService;