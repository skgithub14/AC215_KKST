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
    const [prediction, setPrediction] = useState(null);

    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers
    const handleImageUploadClick = () => {
        inputFile.current.click();
        setPrediction(null);
    }
    const handleOnChange = (event) => {
        console.log(event.target.files);
        setImage(URL.createObjectURL(event.target.files[0]));

        var formData = new FormData();
        formData.append("file", event.target.files[0]);
        DataService.Predict(formData)
            .then(function (response) {
                console.log(response.data);
                setPrediction(response.data);
            })
    }

    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="md" className={classes.container}>
                    {prediction &&
                        <Typography variant="h4" gutterBottom align='center'>
                            {<span className={classes.caption}>Caption: {prediction["caption"]}</span>}
                        </Typography>
                    }
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
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Home);