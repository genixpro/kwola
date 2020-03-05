import styled from 'styled-components';

const FormsMainWrapper = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
`;

const FormsComponentWrapper = styled.div`
  background: #f5f5f5;
  width: 50%;
  @media (max-width: 767px) {
    width: 100%;
  }
  @media only screen and (max-width: 1099px) and (min-width: 768px) {
    width: 80%;
  }
  .mainFormsWrapper {
    * {
      box-sizing: border-box;
    }
    input {
      box-sizing: content-box !important;
    }
    input:-webkit-autofill {
      -webkit-box-shadow: 0 0 0px 1000px #fff inset;
    }
    .mainFormsInfoWrapper {
      width: 100%;
      display: flex;
      flex-direction: row;
      flex-flow: wrap;
      .mainFormsInfoField {
        width: 100%;
        margin-bottom: 20px;

        label {
        }

        > div {
          width: 100%;
          > div > div {
            input {
              height: 24px;
            }
          }
        }
      }
    }

    .mateFormsCheckList {
      display: flex;
      flex-direction: column;
      margin-top: 20px;
      .radiButtonHeader {
        margin: 0;
        font-weight: 400;
        color: rgba(0, 0, 0, 0.54);
        padding: 0;
        font-size: 13px;
        font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;
        line-height: 1;
      }
      .mateFormsRadioList {
        > div {
          display: flex;
          flex-direction: row;
          .mateFormsRadio {
            margin-right: 30px;
          }
        }
      }
    }
    .mateFormsFooter {
      display: flex;
      flex-direction: column;
      margin-top: 15px;
      .mateFormsChecBoxList {
        margin-bottom: 10px;
      }

      .mateFormsSubmit {
        display: flex;
        flex-direction: row;
        .mateFormsSubmitBtn {
          background: #3f51b5;
          color: #fff;
          margin-right: 15px;
          box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
            0px 2px 2px 0px rgba(0, 0, 0, 0.14),
            0px 3px 1px -2px rgba(0, 0, 0, 0.12);
        }
        .mateFormsClearBtn {
        }
      }
    }
  }
`;

export { FormsComponentWrapper, FormsMainWrapper };
