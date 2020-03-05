import styled from "styled-components";
import { palette } from "styled-theme";
import WithDirection from "../../../settings/withDirection";
import Papers from "../../uielements/paper";
import { PageTitle } from "../paperTitle/paperTitle.style";

const paperContent = styled.div`
  padding: 40px 30px;

  @media only screen and (max-width: 767px) {
    padding: 30px 15px;
  }

  &.nopadding {
    padding: 0;
  }

  &.scrolling {
    overflow-x: auto;
  }

  > p {
    font-size: 14px;
    color: ${palette("grey", 9)};
    font-weight: inherit;
    line-height: 1.5;
    margin: 0 0 25px;
    text-align: ${props => (props["data-rtl"] === "rtl" ? "right" : "left")};
  }
`;

const ContListItems = styled.li`
  font-size: 14px;
  color: ${palette("grey", 9)};
  font-weight: inherit;
  line-height: 1.5;
  margin-bottom: 5px;
  position: relative;
  padding-left: 44px;

  &:before {
    content: "";
    width: 4px;
    height: 4px;
    background-color: ${palette("grey", 9)};
    display: inline-block;
    margin: 0;
    transform: scale(0.8);
    border-radius: 50%;
    position: absolute;
    top: 9px;
    left: 20px;
  }
`;

const ContList = styled.ul`
  width: 100%;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  margin-bottom: 5px;
`;

const DemoWrappers = styled.div`
  width: 100%;
  padding: 35px;
  background-color: ${palette("grey", 12)};
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;

  @media only screen and (max-width: 767px) {
    padding: 35px 15px;
    overflow: hidden;
    overflow-x: auto;
  }

  &.transparent {
    background-color: transparent;
    ${"" /* display: block; */} display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: flex-start;
    padding: 0;
    margin-top: 35px;
  }

  &.left {
    align-items: flex-start;
    justify-content: flex-start;
  }

  &.center {
    align-items: center;
    justify-content: center;
  }

  &.right {
    align-items: flex-end;
    justify-content: flex-end;
  }

  &.column {
    flex-direction: column;
  }
`;

const Codes = styled.span`
  font-size: 13px;
  color: ${palette("grey", 9)};
  font-weight: inherit;
  line-height: 1;
  padding: 5px 10px;
  background: ${palette("grey", 2)};
  border-radius: 12px;
  display: inline-block;
`;

const Papersheet = styled(Papers)`
  &.noShadow {
    background-color: transparent;
    box-shadow: none;

    ${PageTitle} {
      padding: 30px 0;
    }

    ${paperContent} {
      padding: 30px 0;
    }
  }

  &.stretched {
    height: 100%;
    display: flex;
    flex-direction: column;

    ${paperContent} {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    ${DemoWrappers} {
      height: 100%;
    }
  }
`;

const Contents = WithDirection(paperContent);

export { Contents, ContList, ContListItems, DemoWrappers, paperContent, Codes };

export default Papersheet;
