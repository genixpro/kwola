import styled from 'styled-components';
import { palette } from 'styled-theme';
import WithDirection from '../../settings/withDirection';
import TextFields from '../../components/uielements/textfield';
import Checkboxs from '../../components/uielements/checkbox';
import Dialogs, {
  DialogTitle as DialogTitles,
} from '../../components/uielements/dialogs';

const FirebaseLoginModal = styled.form`
  padding: 50px;
  padding-top: 25px;
  padding-bottom: 25px;
  * {
    box-sizing: border-box;
  }

  input:-webkit-autofill {
    -webkit-box-shadow: 0 0 0px 1000px #fff inset;
  }
  input {
    box-sizing: content-box !important;
    &::-webkit-input-placeholder {
      color: ${palette('grayscale', 0)};
    }

    &:-moz-placeholder {
      color: ${palette('grayscale', 0)};
    }

    &::-moz-placeholder {
      color: ${palette('grayscale', 0)};
    }
    &:-ms-input-placeholder {
      color: ${palette('grayscale', 0)};
    }
  }
  .resetPass {
    box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
      0px 2px 2px 0px rgba(0, 0, 0, 0.14), 0px 3px 1px -2px rgba(0, 0, 0, 0.12);
  }
  .firebaseCancelBtn {
    padding: 0 15px;
    font-size: 14px;
    border-radius: 3px;
    min-height: 32px;
    border: 1px solid #bababa;
    font-size: 12px;
    box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
      0px 2px 2px 0px rgba(0, 0, 0, 0.14), 0px 3px 1px -2px rgba(0, 0, 0, 0.12);
  }
  .firebaseloginBtn {
    color: #fff;
    background-color: #108ee9;
    border-color: #108ee9;
    margin-left: 8px;
    padding: 0 15px;
    font-size: 14px;
    border-radius: 3px;
    min-height: 33px;
    font-size: 12px;
    box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
      0px 2px 2px 0px rgba(0, 0, 0, 0.14), 0px 3px 1px -2px rgba(0, 0, 0, 0.12);
    &:hover {
      background-color: #49a9ee;
      border-color: #49a9ee;
    }
  }

  .resetPass {
    border: 1px solid #bababa;
    min-width: auto;
    min-height: 32px;
    padding: 0 15px;
    border-radius: 4px;
    margin-right: 8px;
    font-size: 12px;
    @media (max-width: 500px) {
      margin-bottom: 10px;
    }
  }
`;
const TextField = styled(TextFields)`
  width: 100%;
  > div {
    &:before {
      background-color: #bababa;
    }
  }
`;
const Checkbox = styled(Checkboxs)``;
const Dialog = styled(Dialogs)``;
const DialogTitle = styled(DialogTitles)`
  padding-left: 50px;
  padding-bottom: 0;
  .form-dialog-title-text {
    font-size: 14px;
    line-height: 21px;
    font-weight: 500;
    color: rgba(0, 0, 0, 0.85);
  }
`;
export default WithDirection(FirebaseLoginModal);
export { TextField, Checkbox, Dialog, DialogTitle };
