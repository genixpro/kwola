import styled from 'styled-components';
import { palette } from 'styled-theme';
import Buttons from '../../uielements/button';
import Icons from '../../uielements/icon';
import IconButtons from '../../uielements/iconbutton';

const Icon = styled(Icons)`
  font-size: 20px;
  color: ${palette('grey', 7)};
`;

const IconButton = styled(IconButtons)`
  width: 30px;
  height: 30px;
  padding: 0;
`;

const Button = styled(Buttons)`
  background-color: ${palette('blue', 6)};
  color: #ffffff;

  &:hover {
    background-color: ${palette('blue', 7)};
  }

  > span {
    span {
      font-weight: 700;
    }

    &:last-child {
      span {
        background-color: #ffffff;
      }
    }
  }
`;

const EditorWrapper = styled.div`
  position: relative;
  padding: 5px 20px 0;
`;

const ComposeForm = styled.div`
  padding: 0 0 20px;
  text-align: left;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;

  * {
    box-sizing: border-box;
  }

  .suBj3cT {
    height: 44px;
    margin-top: 10px;
    display: flex;
    align-items: center;

    input {
      font-size: 17px;
      font-weight: 700;
      color: ${palette('grey', 8)};
      padding: 0 20px;

      &::-webkit-input-placeholder {
        opacity: 1;
        color: ${palette('grey', 6)};
      }

      &:-moz-placeholder {
        opacity: 1;
        color: ${palette('grey', 6)};
      }

      &::-moz-placeholder {
        opacity: 1;
        color: ${palette('grey', 6)};
      }
      &:-ms-input-placeholder {
        opacity: 1;
        color: ${palette('grey', 6)};
      }
    }
  }

  .mailComposeEditor {
    .quill {
      width: 100%;
      display: flex;
      flex-direction: column;

      .ql-container {
        border: 0;
        min-height: 220px;
        margin: 10px 0;

        .ql-editor {
          min-height: 50vh;
          max-height: 55vh;
          padding: 5px 0 10px;

          p,
          h1,
          h2,
          h3,
          h4,
          h5,
          h6,
          span {
            color: ${palette('grey', 9)};
          }
        }
      }

      .ql-toolbar {
        display: inline-flex;
        flex-wrap: wrap;
        order: 1;
        border: 0;
        width: calc(100% - 70px);
        padding: 4px 8px;
        margin-left: auto;

        @media only screen and (max-width: 770px) {
          width: 100%;
          padding: 4px 0 15px;
        }

        .ql-picker-options {
          position: fixed;
          z-index: 100000;
          top: auto;
          width: auto;
          left: auto;
          right: auto;
          min-width: 100px;
        }
      }
    }
  }

  .sEnd-bTn {
    position: absolute;
    bottom: 0;
    left: 20px;

    @media only screen and (max-width: 770px) {
      position: relative;
      bottom: auto;
      left: auto;
    }
  }

  .dEl3Te-bTn {
    position: absolute;
    bottom: 0;
    right: 20px;

    @media only screen and (max-width: 770px) {
      position: relative;
      bottom: auto;
      right: auto;
      display: inline-flex;
      float: right;
    }

    @media only screen and (max-width: 780px) {
      right: 0;
    }
  }
`;

export { EditorWrapper, Button, IconButton, Icon };
export default ComposeForm;
