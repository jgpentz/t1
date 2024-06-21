import * as React from "react"

interface FrfLogoProps {
    width: number
    height: number
}

const FrfLogo = ({width, height}: FrfLogoProps) => {
    // Calculate padding to ensure content remains fully visible
    const padding = 7; // You can adjust this value as needed

    // Calculate viewBox dimensions with padding
    const viewBoxWidth = width;
    const viewBoxHeight = height;
    const viewBox = `15 0 ${viewBoxWidth + 25} ${viewBoxHeight + 25}`;

    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width={viewBoxWidth}
            height={viewBoxHeight}
            viewBox={viewBox}
        >
            <g
            style={{
                display: "inline",
            }}
            >
            <path
                d="M13.93 6.302s-20.822 26.154-.508 55.693c0 0-15.696-26.38.508-55.693z"
                style={{
                fill: "#000",
                fillOpacity: 1,
                stroke: "#000",
                strokeWidth: ".16757828px",
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                }}
            />
            <path
                d="M19.959 11.223S6.579 31.93 19.48 57.247c-17.356-25.077.479-46.024.479-46.024z"
                style={{
                fill: "#000",
                fillOpacity: 1,
                stroke: "#000",
                strokeWidth: ".16757828px",
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                }}
            />
            <path
                d="M24.256 52.939s-14.05-16.59.17-37.58c-9.82 19.89-.17 37.58-.17 37.58z"
                style={{
                fill: "#000",
                fillOpacity: 1,
                stroke: "#000",
                strokeWidth: ".16757828px",
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                }}
            />
            <path
                d="M28.826 49.807s-7.448-13.711-.169-31.232c-11.172 17.182.17 31.232.17 31.232z"
                style={{
                fill: "#000",
                fillOpacity: 1,
                stroke: "#000",
                strokeWidth: ".16757828px",
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                }}
            />
            <path
                d="M32.889 46.337s-5.67-11.088 0-24.292c-8.887 14.304 0 24.292 0 24.292z"
                style={{
                fill: "#000",
                fillOpacity: 1,
                stroke: "#000",
                strokeWidth: ".16757828px",
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                }}
            />
            <path
                d="M93.433 40.215H87.68v5.804h5.753V44.05l2.134.442v1.085q0 1.258-.477 1.718-.469.451-1.796.451h-5.475q-1.336 0-1.804-.451-.469-.46-.469-1.718v-4.833q0-1.249.469-1.709.468-.46 1.804-.46h5.475q1.327 0 1.796.46.477.46.477 1.71v.867l-2.134.338zM100.946 46.306h6.056v-4.694h-6.056zm-1.83-4.304q0-1.023.399-1.396.399-.374 1.553-.374h5.813q1.162 0 1.553.374.399.373.399 1.396v3.965q0 1.033-.408 1.406-.4.373-1.544.373h-5.813q-1.154 0-1.553-.373-.4-.373-.4-1.406zM117.986 41.525h-4.182v2.083h4.182q.477 0 .65-.14.183-.138.183-.468v-.867q0-.33-.182-.469-.174-.139-.651-.139zm-5.987 6.221v-7.514h6.768q.928 0 1.345.348.416.338.416 1.11v1.77q0 .764-.416 1.102-.417.338-1.345.338h-1.033l3.575 2.846h-2.629l-3.15-2.846h-1.726v2.846zM123.556 47.746v-7.514h6.767q.937 0 1.345.348.417.338.417 1.11v1.666q0 .772-.417 1.12-.416.337-1.345.337h-4.962v2.933zm5.987-6.22h-4.182v1.995h4.182q.477 0 .65-.14.183-.138.183-.468v-.78q0-.33-.183-.469-.173-.139-.65-.139zM136.38 46.306h6.055v-4.694h-6.056zm-1.831-4.304q0-1.023.399-1.396.399-.374 1.553-.374h5.813q1.162 0 1.553.374.399.373.399 1.396v3.965q0 1.033-.408 1.406-.399.373-1.544.373H136.5q-1.154 0-1.553-.373-.4-.373-.4-1.406zM153.42 41.525h-4.183v2.083h4.182q.477 0 .651-.14.182-.138.182-.468v-.867q0-.33-.182-.469-.174-.139-.65-.139zm-5.987 6.221v-7.514h6.767q.928 0 1.345.348.416.338.416 1.11v1.77q0 .764-.416 1.102-.417.338-1.345.338h-1.032l3.574 2.846h-2.629l-3.15-2.846h-1.726v2.846zM157.22 47.746l4.459-7.514h1.761l4.53 7.514h-2.057l-.946-1.648h-4.971l-.946 1.648zm3.557-3.02h3.444l-1.753-3.131zM173.357 41.577v6.169h-1.83v-6.169H167.9v-1.345h9.092v1.345zM179.36 47.746v-7.514h1.823v7.514zM186.18 46.306h6.056v-4.694h-6.056zm-1.83-4.304q0-1.023.399-1.396.399-.374 1.553-.374h5.813q1.162 0 1.553.374.399.373.399 1.396v3.965q0 1.033-.408 1.406-.399.373-1.544.373h-5.813q-1.154 0-1.553-.373-.4-.373-.4-1.406zM197.234 47.746v-7.514h1.258l6.377 4.99v-4.99h1.657v7.514h-1.267l-6.36-5.067v5.067z"
                style={{
                fontStyle: "normal",
                fontVariant: "normal",
                fontWeight: 400,
                fontStretch: "semi-expanded",
                fontSize: "medium",
                lineHeight: "0%",
                fontFamily: "&quot",
                fontVariantLigatures: "normal",
                fontVariantCaps: "normal",
                fontVariantNumeric: "normal",
                fontFeatureSettings: "normal",
                textAlign: "start",
                letterSpacing: 0,
                wordSpacing: 0,
                writingMode: "horizontal-tb",
                textAnchor: "start",
                fill: "#000",
                fillOpacity: 1,
                stroke: "none",
                strokeWidth: 0.74036193,
                }}
            />
            <path
                d="M36.526 24.556v19.329s8.552-5.93 16.339-7.876l13.066-.084a3.578 3.578 0 0 0 4.844 1.361c1.727-.957 2.403-3.08 1.445-4.807a3.58 3.58 0 0 0-4.865-1.393c-.572.317-1.092.854-1.419 1.424l-13.071-.083c-7.787-1.947-16.339-7.871-16.339-7.871Z"
                style={{
                fill: "#000",
                fillOpacity: 1,
                fillRule: "evenodd",
                stroke: "#000",
                strokeWidth: ".16757828px",
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                markerStart: "none",
                }}
            />
            <path
                d="M85.737 29.636v-9.17h9.275v1.604h-7.14v2.412h3.86v1.675h-3.86v3.479zM97.675 29.636v-9.17h2.135v9.17zM111.41 22.053h-4.798v2.55h4.798q.555 0 .754-.156.209-.164.209-.572v-1.102q0-.4-.209-.555-.2-.165-.754-.165zm-6.898 7.583v-9.17h7.843q1.093 0 1.58.424.494.417.494 1.354v2.151q0 .929-.495 1.354-.486.425-1.579.425h-1.197l4.234 3.462h-3.063l-3.696-3.462h-2.021v3.462zM126.723 22.07h-6.004v1.97h5.718q1.344 0 1.813.46.477.46.477 1.709v1.258q0 1.25-.477 1.709-.469.46-1.813.46h-5.666q-1.345 0-1.822-.46-.468-.46-.468-1.71v-.25l1.882-.391v1.084h6.481v-2.09h-5.717q-1.336 0-1.805-.46-.468-.46-.468-1.71v-1.015q0-1.25.468-1.71.469-.459 1.805-.459h5.205q1.293 0 1.779.443.495.442.495 1.57v.19l-1.883.443zM137.412 22.105v7.53h-2.134v-7.53h-4.208v-1.64h10.567v1.64zM158.018 22.053h-4.798v2.55h4.798q.555 0 .754-.156.209-.164.209-.572v-1.102q0-.4-.209-.555-.2-.165-.754-.165zm-6.898 7.583v-9.17h7.843q1.094 0 1.58.424.494.417.494 1.354v2.151q0 .929-.495 1.354-.485.425-1.579.425h-1.197L162 29.636h-3.063l-3.696-3.462h-2.021v3.462zM165.384 29.636v-9.17h9.274v1.604h-7.14v2.412h3.86v1.675h-3.86v3.479z"
                style={{
                fontStyle: "normal",
                fontVariant: "normal",
                fontWeight: 400,
                fontStretch: "semi-expanded",
                fontSize: "medium",
                lineHeight: "0%",
                fontFamily: "&quot",
                fontVariantLigatures: "normal",
                fontVariantCaps: "normal",
                fontVariantNumeric: "normal",
                fontFeatureSettings: "normal",
                textAlign: "start",
                letterSpacing: 0,
                wordSpacing: 0,
                writingMode: "horizontal-tb",
                textAnchor: "start",
                fill: "#000",
                fillOpacity: 1,
                stroke: "none",
                strokeWidth: 0.74036193,
                }}
            />
            <g
                aria-label="TM"
                style={{
                fontStyle: "normal",
                fontVariant: "normal",
                fontWeight: 400,
                fontStretch: "normal",
                fontSize: "5.33333349px",
                lineHeight: "125%",
                fontFamily: "Sans",
                fontVariantLigatures: "normal",
                fontVariantCaps: "normal",
                fontVariantNumeric: "normal",
                fontFeatureSettings: "normal",
                textAlign: "start",
                letterSpacing: 0,
                wordSpacing: 0,
                writingMode: "horizontal-tb",
                textAnchor: "start",
                fill: "#000",
                fillOpacity: 1,
                stroke: "none",
                strokeWidth: 1,
                strokeLinecap: "butt",
                strokeLinejoin: "miter",
                strokeOpacity: 1,
                }}
            >
                <path
                d="M429.001 241.109h-1.385v3.419h-.516v-3.42h-1.385v-.458H429zM432.98 244.528h-.515v-3.341l-1.078 2.273h-.308l-1.07-2.273v3.34h-.482v-3.877h.703l1.034 2.16 1-2.16h.716z"
                transform="translate(-248.93 -137.375) scale(.74036)"
                />
            </g>
            </g>
        </svg>
    )
}
export default FrfLogo

