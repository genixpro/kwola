import React from "react";
import Title from "../paperTitle";
import Scrollbar from "../customScrollBar.js";
import Papersheet, {
  Contents,
  ContList,
  ContListItems,
  DemoWrappers,
  Codes
} from "./papersheet.style";

const Code = props => <Codes>{props.children}</Codes>;

const BulletListItem = props => <ContListItems>{props.children}</ContListItems>;
const ContentList = props => <ContList>{props.children}</ContList>;

const DemoWrapper = props => (
  <DemoWrappers
    className={`${props[`data-transparent`] ? "transparent" : ""} ${
      props[`data-align`] === "left"
        ? "left"
        : props[`data-align`] === "right"
          ? "right"
          : props[`data-align`] === "center" ? "center" : ""
    } ${props[`data-direction`] === "column" ? "column" : ""}
			${props.className}`}
  >
    {props.children}
  </DemoWrappers>
);

const Content = props => (
  <Contents
    className={`${props[`no-padding`] ? "nopadding" : ""} ${
      props.className ? props.className : ""
    }`}
  >
    {props.scroll ? (
      <Scrollbar style={{ overflowY: "hidden" }}>{props.children}</Scrollbar>
    ) : (
      props.children
    )}
  </Contents>
);

export default props => (
  <Papersheet
    elevation={props.elevation ? props.elevation : 1}
    className={`${props[`data-noshadow`] ? "noShadow" : ""} ${
      props.stretched ? "stretched" : ""
    } ${props.className ? props.className : ""}
    `}
    style={props.style}
    onClick={() => {
      if (props.onClick) {
        props.onClick();
      }
    }}
  >
    {props.title ? <Title title={props.title} subtitle={props.subtitle} /> : ""}
    <Content {...props}>{props.children}</Content>
  </Papersheet>
);
export { DemoWrapper, Code, ContentList, BulletListItem, Content };
