import styled from 'styled-components';
import { palette } from 'styled-theme';
import Dialog from '../uielements/dialogs';
import WithDirection from '../../settings/withDirection';

const DialogWrapper = styled(Dialog)`
  > div {
    &:last-child {
      max-width: 520px;
      box-sizing: border-box;
      margin: auto 20px;
    }
  }

  .modalTitleWrapper {
    padding: 15px 16px;
    background: ${palette('grey', 0)};
    color: ${palette('grey', 9)};
    border-bottom: 1px solid ${palette('grey', 3)};
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-orient: horizontal;
    -webkit-box-direction: normal;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-box-pack: justify;
    -ms-flex-pack: justify;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
    position: relative;

    .modalTitle {
      font-size: 15px;
      color: ${palette('grey', 9)};
    }

    button.modalCloseBtn {
      width: 32px;
      height: 32px;
      position: absolute;
      top: 50%;
      right: 10px;
      transition: all 0.3s ease;
      margin-top: -16px;

      span.material-icons {
        font-size: 16px;
      }

      &:hover {
        span.material-icons {
          color: ${palette('grey', 9)};
        }
      }
    }
  }
`;

const ButtonWrapper = styled.div`
  width: 100%;
  float: right;
  text-align: right;
  box-sizing: border-box;
  padding: 10px 16px 20px;
  display: flex;
  justify-content: flex-end;
  align-items: center;

  button {
    margin-left: 10px;
  }
`;

const CardInfoWrapper = styled.div`
  margin-top: 20px;
`;

const WDInfoFormWrapper = styled.div`
  .cardInfoForm {
    display: flex;
    width: 100%;
    padding: 16px;
    box-sizing: border-box;
    flex-wrap: wrap;

    .cardInput {
      width: 100%;
      margin-bottom: 10px;

      label {
        font-size: 0.9rem;
        color: ${palette('grey', 8)};
      }

      > div {
        input {
          height: 1.2em;
        }
      }

      &.notes {
        order: 2;
        margin-bottom: 0;
      }

      &.expiry,
      &.cvc {
        width: calc(100% / 2 - 5px);
      }

      &.expiry {
        color: #000000;
        margin: ${props =>
          props['data-rtl'] === 'rtl' ? '0 0 10px 10px' : '0 10px 10px 0'};
      }

      &::-webkit-input-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 0)};
      }

      &:-moz-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 0)};
      }

      &::-moz-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 0)};
      }
      &:-ms-input-placeholder {
        text-align: ${props =>
          props['data-rtl'] === 'rtl' ? 'right' : 'left'};
        color: ${palette('grey', 0)};
      }
    }
  }
`;

const InfoFormWrapper = WithDirection(WDInfoFormWrapper);

export { ButtonWrapper, CardInfoWrapper, InfoFormWrapper, DialogWrapper };
