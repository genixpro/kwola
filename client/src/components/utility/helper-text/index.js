import React from "react";
import HelperText from "./style";

export default ({ text = "", className = "helperText", imgSrc = "" }) => (
  <HelperText className={className}>
    {imgSrc !== "" ? <img alt="#" src={imgSrc} /> : ""}
    <h3>{text}</h3>
  </HelperText>
);
