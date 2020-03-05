import styled from 'styled-components';

export const Wrapper = styled.div`
  font-size: 16px;
  line-height: 30px;
  color: #2d3446;
  font-family: 'Roboto';
  font-weight: 500;
  && label {
    font-size: 16px;
    line-height: 30px;
    color: #2d3446;
    font-family: 'Roboto';
    font-weight: 500;
  }
  .field-container {
    display: flex;
    align-items: center;
    color: #1890ff;
    font-size: 16px;
    font-weight: 500;
    margin: 30px 0 50px;
  }
  .MuiButton-containedPrimary {
    width: 100%;
  }
  .MuiFormControlLabel-root {
    .MuiFormControlLabel-label {
      font-weight: 500;
    }
  }
`;
