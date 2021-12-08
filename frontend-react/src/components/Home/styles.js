
const styles = theme => ({
    root: {
        flexGrow: 1,
        minHeight: "100vh"
    },
    grow: {
        flexGrow: 1,
    },
    main: {

    },
    container: {
        backgroundColor: "#ffffff",
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    menu_block:{
        textAlign: "center",
    },
    select_title: {
        position: 'relative',
        fontSize: "22px",
        fontWeight: 'bold'
    },
    menu: {
        position: 'relative',
        fontSize: "20px",
    },
    dropzone: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        margin: "20px",
        borderWidth: "2px",
        borderRadius: "2px",
        borderColor: "#cccccc",
        borderStyle: "dashed",
        backgroundColor: "#fafafa",
        outline: "none",
        transition: "border .24s ease-in-out",
        cursor: "pointer",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        minHeight: "400px",
    },
    fileInput: {
        display: "none",
    },
    button_block:{
        textAlign: "center"
    },
    button: {
        position: 'relative',
        fontSize: "20px",
    },
    preview: {
        width: "100%",
    },
    help: {
        color: "#302f2f",
        fontSize: "20px"
    },
    caption: {
        color: "#000000",
    }
});

export default styles;