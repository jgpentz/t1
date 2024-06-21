import { ActionIcon, Box, Collapse, Grid, Group, Text, TextInput, Tooltip, UnstyledButton } from "@mantine/core";
import { Dispatch, SetStateAction, useEffect, useRef, useState } from "react"
import classes from './FileOptions.module.css'; // Import your CSS module for styling
import { TbBorderStyle, TbBorderStyle2, TbChevronDown, TbChevronRight, TbEye, TbEyeClosed, TbMathXDivideY, TbMathXDivideY2, TbTrash, TbTriangle } from "react-icons/tb";
import { SGraphDataLiteral, SparamFiles } from "@/pages/Sparams.page";

// Interface to define what an sparam should contain
interface sparam {
    ports: string;
}

// Interface to define the props for the FileOptions component
interface FileOptionsProps {
    sparams: Record<string, SparamFiles>;
    setSparams: Dispatch<SetStateAction<Record<string, SparamFiles>>>;
    fname: string;
    snames: string[];
}

export default function FileOptions({ sparams, setSparams, fname, snames }: FileOptionsProps) {
    const [opened, setOpened] = useState(true);
    const [visible, setVisible] = useState(true);
    const [showTooltip, setShowTooltip] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [text, setText] = useState(fname);
    const inputRef = useRef<HTMLInputElement>(null);
    const textRef = useRef<HTMLParagraphElement>(null);

    const handleTextClick = () => {
        setIsEditing(true);
        setTimeout(() => inputRef.current?.focus(), 0); // Focus the input field when it appears
    };

    const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setText(event.target.value);
    };

    const handleBlur = () => {
        setIsEditing(false);
        setSparams((prevSparams) => {
            const updatedSparams = { ...prevSparams };
            const currentData = updatedSparams[fname];

            // Update the name key-value pair for each key that starts with 's'
            for (const key in currentData) {
                if (key.startsWith('s')) {
                    (currentData[key] as any).name = text + " " + key;  // Set the name to the new text
                }
            }

            delete updatedSparams[fname];
            updatedSparams[text] = currentData;

            return updatedSparams
        })
    };


    /* Check if the filename is overflowing so that we can provide a tooltip */
    useEffect(() => {
        const textElement = textRef.current;
        if (textElement) {
            setShowTooltip(textElement?.offsetWidth < textElement?.scrollWidth);
        }
    }, [fname]);

    const deleteFile = (() => {
        // Append the new sparams data
        setSparams(prevSparams => {
            const { [fname]: _, ...newSparams } = prevSparams;
            return newSparams; 
        });
    });

    const togglevisible = (sname?: string) => {
        setSparams(prevSparams => {
            const updatedFileData = { ...prevSparams[fname] };

            // Have to cast to unknown first before casting to GraphData Literal
            // because the TypeScript compiler doesn't understand that
            // updatedFileData is of type DataSet.  It thinks it has type SparamFiles
            if (sname) {
                (updatedFileData[sname] as unknown as SGraphDataLiteral).visible = !(updatedFileData[sname] as unknown as SGraphDataLiteral).visible;
            } else {
                setVisible((o) => !o);
                for (const key in updatedFileData) {
                    if (key.startsWith('s')) {
                        // Ensure updatedFileData[key] is treated as a GraphLiteral
                        (updatedFileData[key] as unknown as SGraphDataLiteral).visible = !visible;
                    }
                }
            }
            return { ...prevSparams, [fname]: updatedFileData };
        });
    };

    // Map through sparams to create options for each s param contained in the file
    const items = snames.map((sname, index) => (
        <Grid key={`${sname}-${index}`} className={classes.Sparams} columns={20}>
            <Grid.Col span={3}></Grid.Col>
            {/* Toggle visibility */}
            <Grid.Col span={3}>
                <ActionIcon className={classes.Icon} variant="transparent" onClick={() => togglevisible(sname)}>
                    {sparams[fname] && sparams[fname][sname].visible ? (
                        <TbEye color="black" size="1.5em"/> 
                    ) : (
                        <TbEyeClosed color="black" size="1.5em"/>
                    )}
                </ActionIcon>
            </Grid.Col>
            {/* S param name */}
            <Grid.Col span={5} className={classes.SparamText}>{sname}</Grid.Col>
            {/* Select line style */}
            <Grid.Col span={3}>
                <ActionIcon className={classes.Icon} variant="transparent" onClick={() => setVisible((o) => !o)}>
                    <TbBorderStyle2 color={sparams[fname] && (sparams[fname][sname] as any).color} size="1.5em"/> 
                </ActionIcon>
            </Grid.Col>
            {/* Normalize to this sparam */}
            <Grid.Col span={3}>
                <ActionIcon className={classes.Icon} variant="transparent" onClick={() => setVisible((o) => !o)}>
                    <TbTriangle color="black" size="1.5em"/> 
                </ActionIcon>
            </Grid.Col>
        </Grid>
    ));

    return(
        <>
            {/* Header section of the FileOptions component */}
            <Grid className={classes.Header} columns={20}>
                {/* Toggle visibility */}
                <Grid.Col span={3}>
                    <ActionIcon 
                        className={`${classes.Icon} ${classes.HeaderFirstIcon}`} 
                        variant="transparent" 
                        onClick={() => togglevisible()}
                    >
                        {visible ? (
                            <TbEye color="black" size="1.5em"/> 
                        ) : (
                            <TbEyeClosed color="black" size="1.5em"/>
                        )}
                    </ActionIcon>
                </Grid.Col>
                {/* Filename */}
                <Grid.Col span={8}>
                    {showTooltip ? (
                        <Tooltip label={fname} withArrow>
                            {isEditing ? (
                                <TextInput
                                    ref={inputRef}
                                    value={text}
                                    onChange={handleInputChange}
                                    onBlur={handleBlur}
                                    className={classes.HeaderText}
                                />
                            ) : (
                                <Text
                                    ref={textRef}
                                    className={classes.HeaderText}
                                    onClick={handleTextClick}
                                    >
                                    {text}
                                </Text>
                            )}
                        </Tooltip>
                    ) : (
                        isEditing ? (
                            <TextInput
                                ref={inputRef}
                                value={text}
                                onChange={handleInputChange}
                                onBlur={handleBlur}
                                className={classes.HeaderText}
                            />
                        ) : (
                            <Text
                                ref={textRef}
                                className={classes.HeaderText}
                                onClick={handleTextClick}
                                >
                                {text}
                            </Text>
                        )
                    )}
                </Grid.Col>
                {/* Do some math */}
                <Grid.Col span={3}>
                    <ActionIcon className={classes.Icon} variant="transparent" onClick={() => setVisible((o) => !o)}>
                        <TbMathXDivideY2 color="black" size="1.5em"/> 
                    </ActionIcon>
                </Grid.Col>
                {/* Delete this file */}
                <Grid.Col span={3}>
                    <ActionIcon className={classes.Icon} variant="transparent" onClick={() => deleteFile()}>
                        <TbTrash color="black" size="1.5em"/> 
                    </ActionIcon>
                </Grid.Col>
                {/* Collapse/expand the options for this file */}
                <Grid.Col span={3}>
                    <ActionIcon className={classes.Icon} variant="transparent" onClick={() => setOpened((o) => !o)}>
                        {opened ? (
                            <TbChevronDown color="black" size="1.3em"/> 
                        ) : (
                            <TbChevronRight color="black" size="1.3em"/>
                        )}
                    </ActionIcon>
                </Grid.Col>
            </Grid>
            {/* Collapse section that shows/visibles based on the opened state */}
            <Collapse in={opened} className={classes.CollapsedGroup} transitionDuration={0}>
                {items}
            </Collapse>
        </>
    )
}
