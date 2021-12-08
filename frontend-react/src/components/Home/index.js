import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';

import DataService from "../../services/DataService";
import styles from './styles';

const Home = (props) => {
    const { classes } = props;

    console.log("================================== Home ======================================");

    const inputFile = useRef(null);

    // Component States
    const [image, setImage] = useState(null);
    const [imageData, setImageData] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [model_type, setModelType] = useState("1");

    const options = [
        {
          label: "Encoder-decoder transformer",
          value: "1",
        },
        {
          label: "Prefix transformer",
          value: "2",
        },
        {
          label: "RNN with attention (baseline model)",
          value: "3",
        },
      ];
 
    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers
    const handleImageUploadClick = () => {
        inputFile.current.click();
    }
    const handleOnChange = (event) => {
        if (event.target.files.length === 1) {
            setPrediction(null);
            console.log(event.target.files);
            setImage(URL.createObjectURL(event.target.files[0]));

            var formData = new FormData();
            formData.append("file", event.target.files[0]);
            setImageData(formData);
        }
    }

    const handleGenerateClick = () => {
        if (imageData===null) return; 
        setPrediction(null);
        if (model_type==="1") {
            DataService.Predict(imageData)
                .then(function (response) {
                    console.log(response.data);
                    setPrediction(response.data);
                })
            } else if (model_type==="2") {
                DataService.Predict_prefix(imageData)
                .then(function (response) {
                    console.log(response.data);
                    setPrediction(response.data);
                })
            } else if (model_type==="3") {
                DataService.Predict_RNN(imageData)
                .then(function (response) {
                    console.log(response.data);
                    setPrediction(response.data);
                })
            }
        
    }
    
    

    const handleSelect = (event) => {
        console.log("Model selected");
        setModelType(event.target.value);      
    }

    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="md" className={classes.container}>
                <div className={classes.menu_block}>
                    <label className={classes.select_title}>Select model type</label>
                    <br/>
                    <select className={classes.menu} 
                        value={model_type} 
                        onChange={(event) => handleSelect(event)}>
                        {options.map((option) => (
                            <option value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </div>
                    <div className={classes.dropzone} onClick={() => handleImageUploadClick()}>
                        <input
                            type="file"
                            accept="image/*"
                            capture="camera"
                            on
                            autocomplete="off"
                            tabindex="-1"
                            className={classes.fileInput}
                            ref={inputFile}
                            onChange={(event) => handleOnChange(event)}
                        />
                        <div><img className={classes.preview} src={image} /></div>
                        <div className={classes.help}>Click here to upload an image</div>
                    </div>
                    <div className={classes.button_block}>
                        <button className={classes.button} onClick={() => handleGenerateClick()} disabled={!image}>
                            Generate caption
                        </button>
                    </div>
                    <br/>
                    {prediction &&
                        <Typography variant="h4" gutterBottom align='center'>
                            {<span className={classes.caption}>{prediction["caption"]}</span>}
                        </Typography>
                    }
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Home);