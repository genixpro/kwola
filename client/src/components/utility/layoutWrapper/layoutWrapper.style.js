import styled from "styled-components";

const LayoutContentWrapper = styled.div`
  padding: 30px;
  display: flex;
  flex-flow: row wrap;
  ${"" /* overflow: hidden; */} align-items: flex-start;
  box-sizing: border-box;

  @media only screen and (max-width: 767px) {
    padding: 20px;
  }

  @media (max-width: 580px) {
    padding: 15px;
  }
`;

export { LayoutContentWrapper };
