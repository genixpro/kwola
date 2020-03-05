import styled from 'styled-components';

const ViewProfileWrapper = styled.div`
  background: #ffffff;
  border: 1px solid #e9e9e9;
  overflow: auto;
  position: absolute;
  right: 0;
  width: 95%;
  height: calc(100vh - 128px);

  .viewProfileTopBar {
    background: #f3f3f3;
    padding: 23px 20px 22px 30px;
    display: flex;
    justify-content: space-between;
    line-height: 1;

    span {
      margin: 0;
      margin-left: 15px;
      cursor: pointer;
      i {
        font-size: 14px;
        font-weight: normal;
      }
    }
  }
  .viewProfileContent {
    padding: 30px 30px 0;
  }
  .viewProfileImage {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  h1 {
    font-size: 21px;
    text-align: center;

    font-weight: 300;
    margin-bottom: 30px;
    color: #212121;
  }
  img {
    height: 120px;
    border-radius: 6px;
  }
  .viewProfileQuickInfo {
    border-top: 1px solid #eaeaea;
    padding-top: 40px;
  }
  @media only screen and (min-width: 768px) {
    width: 350px;
    margin-right: 30px;
  }
`;

const SingleInfoWrapper = styled.div`
  display: flex;

  .viewProfileTitle {
    width: 35%;
    font-size: 14px;
    font-weight: light;
    color: #424242;
    margin-bottom: 30px;
  }
  .viewProfileValue {
    width: 65%;
    font-weight: 500;
    text-align: right;
    font-size: 14px;
    color: #212121;
    margin-bottom: 30px;
  }
`;

export { ViewProfileWrapper, SingleInfoWrapper };
