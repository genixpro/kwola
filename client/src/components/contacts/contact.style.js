import styled from 'styled-components';
import { palette } from 'styled-theme';
import Avatars from '../uielements/avatars';
import Dialog from '../uielements/dialogs';
import Icons from '../uielements/icon';
import InputSearches from '../uielements/inputSearch/';

const Avatar = styled(Avatars)``;
const ContactViewModal = styled(Dialog)`
  z-index: 1401;
`;

const SingleContactCard = styled.div`
  width: 120px;
  height: 120px;
  margin: 0 10px 10px 0;
  padding: 15px 10px 10px;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  background-color: ${palette('grey', 12)};
  box-sizing: border-box;
  cursor: pointer;

  @media (max-width: 450px) {
    width: calc(50% - 10px);
  }

  &:last-child {
    margin-right: 0;
  }

  ${Avatar} {
    width: 50px;
    height: 50px;
    margin-bottom: 10px;
  }

  h2 {
    font-size: 14px;
    font-weight: 500;
    margin: 0;
    color: ${palette('grey', 8)};
    text-align: center;
  }
`;

const ContactGroupViews = styled.div`
  display: flex;
  width: 100%;
  flex-direction: column;
  margin-bottom: 30px;

  &:last-child {
    margin-bottom: 0;
  }

  .alphabet {
    font-size: 14px;
    font-weight: 700;
    color: ${palette('grey', 8)};
    margin-bottom: 10px;
    margin-top: 0;
  }
`;

const ContactGroupItem = styled.div`
  display: flex;
  width: 100%;
  flex-flow: row wrap;
`;

const PersonNameImg = styled.div`
  margin-right: 30px;
  flex-shrink: 0;

  @media only screen and (max-width: 767px) {
    display: flex;
    margin-right: 0;
    flex-direction: column;
    align-items: center;
    margin-bottom: 30px;
  }

  ${Avatar} {
    width: 90px;
    height: 90px;
    margin-bottom: 10px;
    border-radius: 50%;
    overflow: hidden;
  }

  .inputUpload {
    display: none;
  }

  h2 {
    font-size: 14px;
    font-weight: 500;
    margin: 0;
    color: ${palette('grey', 8)};
    text-align: center;
  }
`;

const ModalView = styled.div`
  display: flex;
  width: 100%;
  flex-direction: column;

  .contactInfo {
    width: 100%;
    margin-bottom: 15px;
    display: flex;

    h6 {
      font-size: 13px;
      font-weight: 500;
      margin: 0;
      margin-right: 15px;
      text-transform: capitalize;
      color: ${palette('grey', 8)};
      width: 80px;
      flex-shrink: 0;
      display: flex;

      &:after {
        content: ':';
        display: flex;
        margin-left: auto;
      }
    }

    span {
      font-size: 13px;
      font-weight: 400;
      margin: 0;
      color: ${palette('grey', 5)};
    }
  }
`;

const ButtonGroup = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  margin-top: 15px;
`;

const IconButton = styled(Icons)`
  font-size: 18px;
  color: ${palette('grey', 5)};
  position: absolute;
  top: 10px;
  right: 10px;
  cursor: pointer;
`;

const ContactModal = styled.form`
  padding: 30px 30px 20px;
  display: flex;
  box-sizing: border-box;

  @media only screen and (max-width: 767px) {
    flex-direction: column;
  }

  &.editView {
    flex-direction: column;

    @media only screen and (max-width: 480px) {
      padding: 30px 15px 20px;
    }

    ${PersonNameImg} {
      margin-right: 0;
      margin-bottom: 20px;
      display: flex;
      justify-content: center;
    }
  }
`;

const InputSearch = styled(InputSearches)`
  width: 100%;
`;

export {
  SingleContactCard,
  ContactViewModal,
  Avatar,
  ContactGroupItem,
  ContactGroupViews,
  ContactModal,
  PersonNameImg,
  ModalView,
  ButtonGroup,
  IconButton,
  InputSearch,
};
